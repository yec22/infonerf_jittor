# Reference: https://github.com/mjmjeong/InfoNeRF
import os, sys, random
import imageio
import configargparse
from tqdm import tqdm, trange
import jittor as jt
import numpy as np
from model.utils import *
from model.network import *
from model.loss import *
from dataloader.blender import *

# set cuda
if jt.has_cuda:
    jt.flags.use_cuda = 1

# set seed
np.random.seed(0)
jt.set_global_seed(0)
random.seed(0)

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

# instantiate NeRF's MLP model. (coarse and fine)
def create_nerf(args):
    
    embed_fn, input_ch = get_embedder(10)
    embeddirs_fn, input_ch_views = get_embedder(4)
    skips = [4]
    
    model = NeRF(input_ch=input_ch, input_ch_views=input_ch_views, skips=skips)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(input_ch=input_ch, input_ch_views=input_ch_views, skips=skips)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # create optimizer
    optimizer = jt.nn.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    # load checkpoints
    ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        if args.ckpt_render_iter is not None:
            ckpt_path = os.path.join(os.path.join(basedir,expname, f'{args.ckpt_render_iter:06d}.tar'))

        ckpt = jt.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'entropy_ray_zvals' : args.entropy,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

# batchify the input and get the network output
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    inputs_flat = jt.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    input_dirs = viewdirs[:,None].expand(inputs.shape)
    input_dirs_flat = jt.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = embeddirs_fn(input_dirs_flat)
    embedded = jt.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = jt.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def train(args):

    # load data
    images, poses, render_poses, hwf, i_split = load_data(args.datadir, args.testskip)
    i_train, i_val, i_test = i_split
    near = 2.0
    far = 6.0
    i_train = np.array(args.train_scene)
    images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:]) # white_background
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    print('i_train:', i_train)
    print('TRAIN views are:', i_train)
    print('TEST views are:', i_test)
    print('VAL views are:', i_val)

    # intrinsics
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if args.render_test:
        render_poses = np.array(poses[i_test])
    elif args.render_train:
        render_poses = np.array(poses[i_train])

    # create log dir
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    # create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    N_rgb = args.N_rand

    # loss function
    if args.entropy:
        N_entropy = args.N_entropy
        fun_entropy_loss = Entropy_Loss(args.N_rand, args.entropy_acc_threshold, args.N_entropy)

    # move data to GPU
    render_poses = jt.float32(render_poses)
    images = jt.float32(images)
    poses = jt.float32(poses)

    N_iters = args.N_iters + 1
    start = start + 1
    print("Begin Training ...")

    # training loop
    for i in trange(start, N_iters):
        # randomly choose one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
            
        rgb_pose = poses[img_i, :3,:4]
        rays_o, rays_d = get_rays(H, W, focal, jt.float32(rgb_pose))  # (H, W, 3)

        if i < args.precrop_iters: # for stable training (warm up)
            dH = int(H//2 * args.precrop_frac)
            dW = int(W//2 * args.precrop_frac)
            coords = jt.stack(
                jt.meshgrid(
                    jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                    jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                ), -1)
            if i == start:
                print(f"[Config] center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")                
        else:
            coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)

        coords = jt.reshape(coords, [-1,2])
        select_inds = np.random.choice(coords.shape[0], size=[N_rgb], replace=False)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        batch_rays = jt.stack([rays_o, rays_d], 0)
        target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3) 
        
        # sampling for unseen rays to compute entropy loss     
        if args.entropy and (args.N_entropy !=0):
            img_i = np.random.choice(len(images))
            target = images[img_i]
            pose = poses[img_i, :3,:4]
            rays_o, rays_d = get_rays(H, W, focal, jt.float32(pose))
            
            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = jt.stack(
                    jt.meshgrid(
                        jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter {args.precrop_iters}")   
            else:
                coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)
            
            coords = jt.reshape(coords, [-1,2])
            select_inds = np.random.choice(coords.shape[0], size=[N_entropy], replace=False)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o_ent = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d_ent = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays_entropy = jt.stack([rays_o_ent, rays_d_ent], 0)
                
        N_rgb = batch_rays.shape[1]
        if args.entropy and (args.N_entropy !=0):
            batch_rays = jt.concat([batch_rays, batch_rays_entropy], 1)
        
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, **render_kwargs_train)
        if args.entropy:
            acc_raw = acc 
            alpha_raw = extras['alpha']

        extras = {x:extras[x][:N_rgb] for x in extras}

        rgb = rgb[:N_rgb, :]
        disp = disp[:N_rgb] 
        acc = acc[:N_rgb]
        
        # compute all the loss
        img_loss = img2mse(rgb, target_s) # fine-network pixel loss (MSE loss)
        entropy_ray_zvals_loss = 0

        if args.entropy:
            entropy_ray_zvals_loss = fun_entropy_loss.compute_loss(alpha_raw, acc_raw) # entropy loss
        
        if args.entropy_end_iter is not None:
            if i > args.entropy_end_iter:
                entropy_ray_zvals_loss = 0
        
        loss = img_loss + args.entropy_ray_lambda * entropy_ray_zvals_loss # total loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras: # coarse-network pixel loss (MSE loss)
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
        
        optimizer.step(loss) # update weight

        # update learning rate
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            jt.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() if render_kwargs_train['network_fn'] is not None else None,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() if render_kwargs_train['network_fine'] is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('saved checkpoints at', path)

        if (i % args.i_testset == 0) and (i > 0) and (len(i_test) > 0):
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            with jt.no_grad():
                rgbs, disps = render_path(jt.float32(poses[i_test]), hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('saved testset ...')

            test_loss = img2mse(jt.float32(rgbs), images[i_test])
            test_psnr = mse2psnr(test_loss)
            test_redefine_psnr = img2psnr(jt.float32(rgbs), images[i_test])

            tqdm.write(f"[TEST] Iter: {i} Loss: {test_loss.item()}  PSNR: {test_psnr.item()} redefine_PSNR: {test_redefine_psnr.item()}")
    
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        global_step += 1

        sys.stdout.flush()

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


if __name__=='__main__':
    parser = configargparse.ArgumentParser()

    # config & data options
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/nerf_synthetic/lego', 
                        help='input data directory')

    # training options
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='number of iters')
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    
    # entropy loss options
    parser.add_argument("--N_entropy", type=int, default=100,
                        help='number of entropy ray')
    parser.add_argument("--entropy", action='store_true',
                        help='using entropy ray loss')
    parser.add_argument("--entropy_end_iter", type=int, default=None,
                        help='end iteratio of entropy')
    parser.add_argument("--entropy_acc_threshold", type=float, default=0.1,
                        help='threshold for acc masking')
    
    # loss weight options
    parser.add_argument("--entropy_ray_lambda", type=float, default=1,
                        help='entropy lambda for ray zvals entropy loss')

    # sampling options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')

    # rendering options
    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--ckpt_render_iter", type=int, default=None, 
                        help='checkpoint iteration')
    parser.add_argument("--render_train", action='store_true', 
                        help='render the train set instead of render_poses path')  
    
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='blender', 
                        help='options: blender')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--fewshot", type=int, default=0, 
            help='if 0 not using fewshot, else: using fewshot')

    # logging options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    
    args = parser.parse_args()
    train(args)