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


    def load_img(self):
        poses_img = [Path(args.datadir) / args.img_dir_name.replace('??', str(i)) 
                   for i in range(args.angles)]
        sample_matrix = self._fix_sample_matrix()

        W, H = self.img_wh
        # use first N_images-1 to train, the LAST is val
        all_rays = []
        all_rgbs = []
        all_poses = []
        all_filtersIdx = []
        ids4shapeTrain = []
        tensor_resizer = T.Resize([H, W], antialias=True)
        for r, row in enumerate(sample_matrix):
            images_degraded = sio.loadmat(poses_img[r])['all_degraded']
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

