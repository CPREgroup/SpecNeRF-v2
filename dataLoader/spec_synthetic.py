import glob
import os
from pathlib import Path
import numpy as np
import torch
from dataLoader.llff import LLFFDataset, normalize, center_poses, get_spiral, get_ray_directions_blender, average_poses,ndc_rays_blender, get_rays
from opt import args
from PIL import Image
from torchvision import transforms as T
import scipy.io as sio

class FAKEDataset(LLFFDataset):

    def __init__(self, datadir, split='train', downsample=1, is_stack=False, hold_every=8):
        super().__init__(datadir, split, downsample, is_stack, hold_every)
        
        self.near_far = [0.0, 1.0] if args.ndc_ray == 1 else [0.01, 6.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]]) if args.ndc_ray == 1 else \
            torch.tensor([[-7.0, -7.0, -5], [7.0, 7.0, 5]])
        # self.scene_bbox = torch.tensor([[-1.67, -1.5, -1.0], [1.67, 1.5, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    
    def read_meta(self):
        poses = np.load(os.path.join(self.root_dir, 'mitsuba_poses.npy'))  # (N_images, 4, 4)

        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses) == args.angles, \
                'Mismatch between number of args.angles and number of poses! Please rerun COLMAP!'

        # Step 1: rescale focal length according to training resolution
        H, W = int(512 / self.downsample), int(512 / self.downsample)  # original intrinsics, same for all images
        self.img_wh = np.array([W, H])
        self.focal = 0.5 * W / np.tan(0.5 * 40)  # original focal length

        # build rendering path
        N_views, N_rots = 60, 1 # 120, 2

        self.poses, self.pose_avg = center_poses(poses[:, :3, :], self.blender2opencv)
        self.render_path = get_spiral(self.poses, None, N_views=N_views, n_rot=N_rots, rads_scale=0.3, focal=self.focal)

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions_blender(H, W, [self.focal] * 2)  # (H, W, 3)

    def load_img(self):
        rays_savePath = Path(args.datadir) / f"rays_idgeo{args.colIdx4RGBTrain}_ndc{args.ndc_ray}_{self.split}_ds{self.downsample}_mtx{os.path.split(args.sample_matrix_dir)[1][:-4]}.pth"
        poses_img = [Path(args.datadir) / args.img_dir_name.replace('??', str(i)) 
                   for i in range(args.angles)]
        sample_matrix = self._fix_sample_matrix()

        W, H = self.img_wh
        # use first N_images-1 to train, the LAST is val
        if os.path.exists(rays_savePath):
            data = torch.load(rays_savePath)
            all_rays, all_rgbs, all_poses, all_filtersIdx, ids4shapeTrain = \
                data['rays'], data['rgbs'], data['poses'], data['filterids'], data['id4geo']
        else:
            all_rays = []
            all_rgbs = []
            all_poses = []
            all_filtersIdx = []
            ids4shapeTrain = []
            tensor_resizer = T.Resize([H, W], antialias=True)
            for r, row in enumerate(sample_matrix):
                images_degraded = sio.loadmat(poses_img[r])['spec']
                for c, aimEle in enumerate(row):
                    if aimEle != 1:
                        continue
                    elif c == args.colIdx4RGBTrain:
                        # it's for geometry training
                        ids4shapeTrain.append(len(all_rays))

                    img = torch.FloatTensor(images_degraded[:, :, c:c+1]).permute(2, 0, 1)
                    c2w = torch.FloatTensor(self.poses[r])

                    if self.downsample != 1.0:
                        img = tensor_resizer(img)

                    img = img.reshape(args.observation_channel, -1).permute(1, 0)  # (h*w, 3) RGB
                    all_rgbs.append(img)

                    rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                    if args.ndc_ray == 1:
                        rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
                    # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                    all_rays.append(torch.cat([rays_o, rays_d], 1))  # (h*w, 6)
                    all_poses.append(torch.LongTensor([[r]]).expand([rays_o.shape[0], -1]))
                    all_filtersIdx.append(torch.LongTensor([[c]]).expand([rays_o.shape[0], -1]))

            torch.save({
                'rgbs': all_rgbs,
                'rays': all_rays,
                'poses': all_poses,
                'filterids': all_filtersIdx,
                'id4geo': ids4shapeTrain
            }, rays_savePath)
        print(f'{len(all_rgbs)} of images are loaded!')

        self.ids4shapeTrain = ids4shapeTrain
        self.raysnum_oneimage = all_rays[0].shape[0]
        if not self.is_stack:
            self.all_rays = torch.cat(all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
            self.all_poses = torch.cat(all_poses, 0)
            self.all_filtersIdx = torch.cat(all_filtersIdx, 0)
        else:
            self.all_rays = torch.stack(all_rays, 0)   # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(all_rgbs, 0).reshape(-1,*self.img_wh[::-1], args.observation_channel)  # (len(self.meta['frames]),h,w,3)
            self.all_poses = torch.stack(all_poses, 0)
            self.all_filtersIdx = torch.stack(all_filtersIdx, 0)

