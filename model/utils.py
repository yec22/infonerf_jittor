import jittor as jt
import numpy as np

img2mse = lambda x, y : jt.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.float32([10.]))
to8b = lambda x : (255. * np.clip(x, 0, 1)).astype(np.uint8) # to save image

# Average PSNR (each image pair)
def img2psnr(x, y):
    image_num = x.size(0)
    mses = ((x - y) ** 2).reshape(image_num, -1).mean(-1)
    
    psnrs = [mse2psnr(mse) for mse in mses]
    psnr = jt.stack(psnrs).mean()
    return psnr


def get_rays(H, W, focal, c2w):
    # sample point in 2D image space
    i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))
    i = i.t()
    j = j.t()
    dirs = jt.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -jt.ones_like(i)], -1)
    # rotate and translate ray directions from camera coordinate to the world coordinate
    rays_d = jt.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


# Hierarchical sampling, Reference: https://github.com/bmild/nerf
def sample_pdf(bins, weights, N_samples, det=False):
    # Get PDF
    weights = weights + 1e-5

    pdf = weights / jt.sum(weights, dim=1, keepdims=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.concat([jt.zeros_like(cdf[...,:1]), cdf], -1)

    # Take uniform samples
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((cdf.shape[-1]-1) * jt.ones_like(inds), inds)
    inds_g = jt.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = jt.where(denom < 1e-5, jt.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

# Transforms model's predictions to meaningful values.
def raw2outputs(raw, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=jt.nn.relu: 1.-jt.exp(-act_fn(raw)*dists)
    
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = jt.concat([dists, jt.float32([1e8]).expand(dists[...,:1].shape)], -1)
    dists = dists * jt.norm(rays_d[...,None,:], dim=-1)

    rgb = jt.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    sigma = jt.nn.relu(raw[...,3]) # [N_rays, N_samples]

    weights = alpha * jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    depth_map = jt.sum(weights * z_vals, -1)
    disp_map = 1.0/jt.maximum(jt.float32(1e-10) * jt.ones_like(depth_map), depth_map / jt.sum(weights, -1))
    acc_map = jt.sum(weights, -1)
    rgb_map = jt.sum(weights[...,None] * rgb, -2)
    rgb_map = rgb_map + (1.0-acc_map[...,None]) # white background
    
    extra_info = {}
    extra_info['alpha'] = alpha
    extra_info['sigma'] = sigma
    extra_info['dists'] = dists

    return rgb_map, disp_map, acc_map, weights, depth_map, extra_info


def sample_test_ray(rays_o, rays_d, viewdirs, network, z_vals, network_query):
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    raw = network_query(pts, viewdirs, network)

    rgb = jt.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    sigma = jt.nn.relu(raw[...,3]) # [N_rays, N_samples]

    rgb_map, disp_map, acc_map, weights, depth_map, others = raw2outputs(raw, z_vals, rays_d)

    return rgb, sigma, depth_map