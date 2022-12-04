# Reference: https://github.com/bmild/nerf
import jittor as jt
import numpy as np
import json, os
import imageio

trans_t = lambda t : jt.float32([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]])

rot_phi = lambda phi : jt.float32([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]])

rot_theta = lambda th : jt.float32([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]])

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = jt.array(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]).astype(np.float32)) @ c2w
    return c2w

def load_data(dir, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(dir, 'transforms_{}.json'.format(s)), 'r') as f:
            metas[s] = json.load(f)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(dir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle = float(meta['camera_angle_x'])
    focal = 0.5 * W / np.tan(0.5 * camera_angle)
    render_poses = jt.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    return imgs, poses, render_poses, [H, W, focal], i_split