import traceback

from pathlib import Path

import numpy as np
import torch
from rawpy._rawpy import ColorSpace
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms as T
import scipy.io as sio

import filesort_int
from .ray_utils import *
import rawpy
from opt import args
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120, N_rots=2):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
    # focal *= 0.3

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 70, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views, N_rots=N_rots)
    return np.stack(render_poses)



spec_near_far = [0.0, 1.0]
spec_scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])  # -+ 1.5, 1.67, 1.0
spec_white_bg = args.white_bkgd



class SPECLLFFDataset(Dataset):
    white = T.ToTensor()(sio.loadmat('./myspecdata/decorner/meanwhite.mat')['data'])
    black = T.ToTensor()(sio.loadmat('./myspecdata/decorner/meanblack.mat')['data'])
    if_preload_posebounds = False
    filters_back = []

    def __init__(self, args, datadir, poseid, split='train', downsample=4, hold_every=8, sample_vect=None):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.poseid = poseid
        self.args = args
        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.downsample = downsample
        if sample_vect is None:
            self.sample_vect = np.array([1] * args.filters)
        else:
            self.sample_vect = sample_vect

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.prepare_filters(SPECLLFFDataset)
        self.preload_posebounds(SPECLLFFDataset, self)
        self.read_meta()
        global spec_scene_bbox, spec_near_far, spec_white_bg
        self.white_bg = spec_white_bg

        #         self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
        SPECLLFFDataset.near_far = spec_near_far
        self.scene_bbox = spec_scene_bbox
        # self.scene_bbox = torch.tensor([[-1.67, -1.5, -1.0], [1.67, 1.5, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        
    @staticmethod
    def prepare_filters(cls):
        if not isinstance(cls.filters_back, list):
            return
        for i in range(0, args.filters + 1):
            fi = torch.FloatTensor(
                sio.loadmat(os.path.join(args.datadir, f'../filters/f_{i}.mat'))['filter']
            )[1:, 1:]

            cls.filters_back.append(fi)
        cls.filters_back = torch.stack(cls.filters_back)


    @staticmethod
    def preload_posebounds(cls, self):
        if cls.if_preload_posebounds:
            return
        else:
            cls.if_preload_posebounds = True

        print('preloaded...')

        poses_bounds = np.load(os.path.join(self.root_dir, '../poses_bounds.npy'))  # (N_images, 17)
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == args.angels, \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        cls.near_fars = poses_bounds[:, -2:]  # (N_images, 2)

        # todo load depth
        if args.batch_depth > 0:
            cls.depth_list = cls.load_colmap_depth(poses, cls.near_fars, Path(self.root_dir).parent, self.downsample)

        croph, cropw = args.crop_hw
        # todo delete below
        poses[:,0, -1] = croph
        poses[:,1, -1] = cropw
        # todo delete above

        # Step 1: rescale focal length according to training resolution
        H, W, cls.focal = poses[0, :, -1]  # original intrinsics, same for all images
        cls.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        cls.focal = [cls.focal * cls.img_wh[0] / W, cls.focal * cls.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        cls.poses, cls.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = cls.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        cls.near_fars /= scale_factor
        cls.poses[..., 3] /= scale_factor

        # distances_from_center = np.linalg.norm(cls.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = cls.img_wh
        cls.directions = get_ray_directions_blender(H, W, cls.focal)  # (H, W, 3)
        cls.W, cls.H = W, H

        if not hasattr(cls, 'render_path'):
            # build rendering path
            N_views, N_rots = args.N_views, 2
            cls.render_path = get_spiral(cls.poses, cls.near_fars, N_views=N_views, rads_scale=0.4, N_rots=N_rots)

        # todo, get depth rays
        if args.batch_depth > 0:
            cls.depth_rays = cls.get_depth_rays(Path(self.root_dir).parent, self.downsample)


    @staticmethod
    def load_colmap_depth(poses, bds_raw, basedir, factor=8, bd_factor=.75):
        croph, cropw = args.crop_hw
        data_file = Path(basedir) / f'colmap_depth_{croph}_{cropw}_{factor}.npy'
        if os.path.exists(data_file):
            data_list = np.load(str(data_file), allow_pickle=True).tolist()
            return data_list

        images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
        points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

        Errs = np.array([point3D.error for point3D in points.values()])
        Err_mean = np.mean(Errs)
        print("Mean Projection Error:", Err_mean)

        # print(bds_raw.shape)
        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)

        H, W, focal = poses[0, :, -1]
        near = np.ndarray.min(bds_raw) * .9 * sc
        far = np.ndarray.max(bds_raw) * 1. * sc
        print('near/far:', near, far)

        data_list = []
        for id_im in range(1, len(images) + 1):
            depth_list = []
            coord_list = []
            weight_list = []
            for i in range(len(images[id_im].xys)):
                point2D = images[id_im].xys[i]  # w h
                if np.abs(point2D[0] - 0.5*W) > cropw * 0.5 or np.abs(point2D[1] - 0.5*H) > croph * 0.5:
                    # outside of the crop border
                    continue
                id_3D = images[id_im].point3D_ids[i]
                if id_3D == -1:
                    continue
                point3D = points[id_3D].xyz
                depth = ((-poses[id_im - 1, :3, 2]).T @ (point3D - poses[id_im - 1, :3, 3])) * sc
                if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                    continue
                err = points[id_3D].error
                weight = 2 * np.exp(-(err / Err_mean) ** 2)
                depth_list.append(depth)
                # re-position the coords relative to the crop size
                coord_list.append((point2D - [0.5*(W-cropw), 0.5*(H-croph)]) / factor)
                weight_list.append(weight)
            if len(depth_list) > 0:
                print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
                data_list.append(
                    {"depth": np.array(depth_list, dtype=np.float32),
                     "coord": np.array(coord_list, dtype=np.float32),
                     "weight": np.array(weight_list, dtype=np.float32)})
            else:
                print(id_im, len(depth_list))
        # json.dump(data_list, open(data_file, "w"))
        np.save(data_file, data_list)
        return data_list


    @classmethod
    def get_depth_rays(cls, basedir, factor):
        croph, cropw = args.crop_hw
        data_file = basedir / f'depth_rays_{croph}_{cropw}_{factor}.npy'
        if os.path.exists(data_file):
            data_list = np.load(str(data_file), allow_pickle=True).tolist()
            return data_list

        data_list = []
        for ind, img_d in enumerate(cls.depth_list):
            coord = torch.from_numpy(img_d['coord'])
            rays_o, rays_d = get_rays_by_coord_np(cls.H, cls.W, cls.focal, cls.poses[ind], coord)
            rays_o, rays_d = ndc_rays_blender(cls.H, cls.W, cls.focal[0], 1.0, rays_o, rays_d)
            rayso_d = torch.cat([rays_o.float(), rays_d.float()], 1)  # (h*w, 6)

            data_list.append(rayso_d)
        np.save(data_file, data_list)
        return data_list


    def prepare_imglist(self):
        i_test = torch.tensor([-1], dtype=torch.int64)  # first one is the test image, bc later on it'll be self-increased
        # print('test idx is ', i_test)
        sample_matrix = 1 - self.sample_vect if self.split != 'train' else self.sample_vect
        self.img_list = torch.from_numpy(np.argwhere(sample_matrix == 1).reshape(-1))


    def read_meta(self):
        # load full resolution image then resize
        self.image_paths = filesort_int.sort_file_int(f'{self.root_dir}/images/*', 'dng')
        croph, cropw = self.args.crop_hw

        self.prepare_imglist()

        self.all_rays = []
        self.all_rgbs = []
        self.all_filters = []
        tensor_resizer = T.Resize([self.H, self.W], antialias=True)
        tensor_cropper = T.CenterCrop((croph, cropw))

        c2w = torch.FloatTensor(self.poses[self.poseid])
        rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
        rays_o, rays_d = ndc_rays_blender(self.H, self.W, self.focal[0], 1.0, rays_o, rays_d)
        rayso_d = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
        # use first N_images-1 to train, the LAST is val
        for i in self.img_list:
            image_path = self.image_paths[i + 1]
            img = self.read_non_raw(image_path)   # c h w [0-1]
            if self.args.lsc == 1:
                img = SPECLLFFDataset.img_correction(img)   # lens shade correction & black level correction
            img = tensor_cropper(img)   # c croph cropw [0-1]
            if self.downsample != 1.0:
                # img = img.resize(self.img_wh, Image.LANCZOS)
                img = tensor_resizer(img)   # c h=croph/downsample w=cropw/downsample [0-1]
            img = img.permute(1, 2, 0).contiguous()  # (h, w, 3) RGB
            if self.split == 'train':
                img = img.reshape(-1, 3)
            self.all_rgbs.append(img)
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            self.all_filters.append(SPECLLFFDataset.filters_back[i + 1])
            self.all_rays.append(rayso_d)

        # self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h*w, 3)
        # self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        # self.all_filters = SPECLLFFDataset.filters_back[self.img_list + 1] # [num, 28, 28]
        pass


    def read_non_raw(self, image_path):
        is_raw = image_path.endswith('.dng')
        if is_raw:
            try:
                with rawpy.imread(image_path) as raw:
                    rgb = raw.postprocess(user_wb=(1, 1, 1, 1), output_color=ColorSpace.raw,
                                          no_auto_bright=True, output_bps=16, gamma=(1, 1))
            except:
                print(traceback.format_exc())
            img = self.transform(rgb)   # c h w [0~2^16]
        else:
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img, 255.)   # c h w [0-1]

        return img


    def transform(self, img, maxbits_num=65535.):
        return torch.FloatTensor(img / maxbits_num).permute(2, 0, 1)

    @classmethod
    def img_correction(cls, img):
        return torch.clamp_max(torch.clamp_min(img - 0.014, 0.002) / cls.white, 1)
        # return (img - 0.014) / cls.white

    @staticmethod
    def turbulence_1or2dir(shape_tensor):
        bi_pusher = lambda x: 4 / (1 + torch.exp(-10*(x-0.5))) + 1 # [1, 5]
        number = shape_tensor.numel()
        random_num = int(0.68 * number)
        one_num = number - random_num
        dir_amplifier = torch.cat([bi_pusher(torch.rand(random_num)),
                                  torch.ones(one_num)])[torch.randperm(number)].reshape(shape_tensor.shape)

        return dir_amplifier


    def color_lowrank(self, voxel_num, direct_num):
        assert self.split == 'test', 'dataset must be test type'

        '''randomly choose 'voxel_num' numbers of voxels in the [-1, 1] bounding box'''
        voxels = torch.rand(voxel_num, 3)
        voxels_img = voxels.clone()
        voxels_img[:, 1] = 0.998 - voxels_img[:, 1]
        plane_coord = torch.floor(voxels_img[:, :2] * self.img_wh[None, :])
        voxels = voxels * 2 - 1
        # rows_ind = pixels / self.img_wh[0]
        # cols_ind = pixels % self.img_wh[0]
        '''create the normalized ray directions which across the voxels'''
        '''1. find out the corresponding main view direction rays'''
        plane_coord_chain = torch.tensor(plane_coord[:, 1] * self.img_wh[0] + plane_coord[:, 0], dtype=torch.long)
        directions = self.all_rays[0, plane_coord_chain, 3:6]
        '''2. since now we get one voxel corresponds to one direction, but in fact we need it corresponds to lots of
        different directions, so we expand the direction tensor to 'direct_num' times, and add random numbers on it'''
        new_direction = torch.zeros([directions.shape[0] * direct_num, 3])
        new_voxel = torch.zeros_like(new_direction)
        for i in range(0, new_direction.shape[0], direct_num):
            new_voxel[i:i+direct_num] = voxels[int(i / direct_num), :].expand(direct_num, 3).clone()
            new_direction[i:i+direct_num] = directions[int(i / direct_num), :].expand(direct_num, 3).clone()
        new_direction *= SPECLLFFDataset.turbulence_1or2dir(new_direction)
        '''3. now we need to normalize the direction vectors'''
        new_direction = new_direction / torch.norm(new_direction, dim=-1, keepdim=True)

        return new_voxel.to(device), new_direction.to(device)


    def __len__(self):
        return self.all_rays[0].shape[0]
        # W, H = self.img_wh
        # if self.is_stack:
        #     return self.img_num
        # else:
        #     return self.img_num * W * H

    def __getitem__(self, idx):

        # sample = {'rays': self.all_rays[idx],
        #           'rgbs': self.all_rgbs[idx]}
        #
        # return sample
        return None


class RGBfirstLLFFDataset(SPECLLFFDataset):
    def __init__(self, *args, **kargs):
        super(RGBfirstLLFFDataset, self).__init__(*args, **kargs)

    def prepare_imglist(self):
        i_test = torch.tensor([2], dtype=torch.int64)  # first one is the test image, bc later on it'll be self-increased
        # print('test idx is ', i_test)
        self.img_list = i_test




