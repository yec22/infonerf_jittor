import jittor as jt
import numpy as np
import imageio, os
from tqdm import tqdm

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

# get 3D rays for each pixel in 2D image
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

# batchify the input and get the fn output
def batchify(fn, chunk): 
    def ret(inputs):
        return jt.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

# batchify the ray as input
def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])
    all_ret = {k : jt.concat(all_ret[k], 0) for k in all_ret}
    return all_ret

# volumetric rendering.
def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                perturb=0.0,
                N_importance=0,
                network_fine=None,
                sigma_loss=None,
                entropy_ray_zvals=None,
                ):
    
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3]
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = jt.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1]

    # stratified sampling
    t_vals = jt.linspace(0., 1., steps=N_samples)
    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        mids = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = jt.concat([mids, z_vals[...,-1:]], -1)
        lower = jt.concat([z_vals[...,:1], mids], -1)
        # random perturb
        t_rand = jt.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    if network_fn is not None:
        raw = network_query_fn(pts, viewdirs, network_fn)
        rgb_map, disp_map, acc_map, weights, depth_map, others = raw2outputs(raw, z_vals, rays_d)
    else:
        raw = network_query_fn(pts, viewdirs, network_fine)
        rgb_map, disp_map, acc_map, weights, depth_map, others = raw2outputs(raw, z_vals, rays_d)

    # hierarchical sampling
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.0))
        z_samples = z_samples.detach()

        z_vals = jt.float32(np.sort(jt.concat([z_vals, z_samples], -1).numpy(), axis=-1))
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, others = raw2outputs(raw, z_vals, rays_d)

    # return rendering results
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map}
    ret['sigma'] = others['sigma']
    ret['alpha'] = others['alpha']
    ret['z_vals'] = z_vals
    ret['dists'] = others['dists'] 
    ret['xyz'] = jt.sum(weights.unsqueeze(-1)*pts, -2)
    ret['raw'] = raw

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0

    return ret

# render function to get rgb_map, depth_map ...
def render(H, W, focal, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., c2w_staticcam=None, depths=None,
                  **kwargs):
    if c2w is not None:
        # render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    viewdirs = rays_d
    if c2w_staticcam is not None:
        # visualize effect of viewdirs
        rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
    viewdirs = viewdirs / jt.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = jt.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape

    # create ray batch
    rays_o = jt.reshape(rays_o, [-1,3]).float()
    rays_d = jt.reshape(rays_d, [-1,3]).float()

    near, far = near * jt.ones_like(rays_d[...,:1]), far * jt.ones_like(rays_d[...,:1])
    rays = jt.concat([rays_o, rays_d, near, far], -1) # B x 8
    if depths is not None:
        rays = jt.concat([rays, depths.reshape(-1,1)], -1)
    rays = jt.concat([rays, viewdirs], -1)

    # render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = jt.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

# render a sequence of images given by specific render poses (a render path)
def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None):
    H, W, focal = hwf
    rgbs = []
    disps = []
    accs = []
    
    for i, c2w in enumerate(tqdm(render_poses)):
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.numpy())
        disps.append(disp.numpy())
        accs.append(acc.numpy())
        
        if savedir is not None:
            rgb8 = to8b(rgb.numpy())
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8) # save rgb images
            depth = depth.numpy()
            depth = (np.clip(depth / 5, 0, 1) * 255.0).astype(np.uint8)
            imageio.imwrite(os.path.join(savedir, '{:03d}_depth.png'.format(i)), depth) # save depth images
        
        # to save memory
        del rgb
        del disp
        del acc
        del extras
        del depth

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps