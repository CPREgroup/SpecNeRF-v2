from pathlib import Path
import traceback
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
import rawpy
from rawpy._rawpy import ColorSpace
from .ray_utils import *
from opt import args
from colmapUtils.read_write_model import read_images_binary, read_points3d_binary
# from colmapUtils.read_write_dense import read_points3d_binary
import scipy.io as sio


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


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120, n_rot=2):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views, N_rots=n_rot)
    return np.stack(render_poses)


def _find_test_sample(mtx):
    mtxcp = mtx[...]
    mtxcp[:, 0] = 1
    zero_inds = np.argwhere(mtxcp == 0)
    choose_zero_inds = np.random.choice(np.array(range(zero_inds.shape[0])), 3).tolist()
    # choose_zero_inds = [[0, 0], [5, 0], [11, 0]]

    sample_mtx = np.zeros_like(mtx)
    for idx in choose_zero_inds:
        sample_mtx[tuple(zero_inds[idx])] = 1

    return sample_mtx

class LLFFDataset:
    white = T.ToTensor()(sio.loadmat('./myspecdata/decorner/meanwhite.mat')['data'])
    black = T.ToTensor()(sio.loadmat('./myspecdata/decorner/meanblack.mat')['data'])
    filters_back = []
    depth_mean = 1

    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = downsample

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.prepare_filters(LLFFDataset)
        self.read_meta() # read poses, also for depth data and rays
        self.load_img() # read images and form rays
        self.white_bg = args.white_bkgd

        #         self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
        self.near_far = [0.0, 1.0] if args.ndc_ray == 1 else [0.01, 6.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]]) if args.ndc_ray == 1 else \
            torch.tensor([[-7.0, -7.0, -5], [7.0, 7.0, 5]])
        # self.scene_bbox = torch.tensor([[-1.67, -1.5, -1.0], [1.67, 1.5, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)


    @staticmethod
    def prepare_filters(cls):
        bandstart = args.band_start_idx
        if not isinstance(cls.filters_back, list):
            return
        for i in range(0, args.filters + 1):
            fi = torch.FloatTensor(
                np.diagonal(sio.loadmat(os.path.join(args.datadir, f'../{args.filters_folder}/f_{i}.mat'))['filter'])
            )[bandstart: bandstart + args.spec_channel]

            cls.filters_back.append(fi)
        cls.filters_back = torch.stack(cls.filters_back)


    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)

        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == args.angles, \
                'Mismatch between number of args.angles and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)

        if args.depth_supervise:
            self.depth_list = self.load_colmap_depth(poses, self.near_fars, 
                                                    Path(self.root_dir), self.downsample)

        croph, cropw = args.crop_hw
        poses[:,0, -1] = croph
        poses[:,1, -1] = cropw

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        N_views, N_rots = 60, 1 # 120, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views, n_rot=N_rots)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

        if args.depth_supervise:
            self.depth_rays = self.get_depth_rays()
            self.combine_depthimages()

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)


    def _fix_sample_matrix(self):
        self.training_matrix = np.hstack((np.array([0] * args.angles)[:, np.newaxis], 
                                          sio.loadmat(args.sample_matrix_dir)['mask'])) # first column is pure rgb image
        self.training_matrix[:, args.colIdx4RGBTrain] = 1 # the column image is used for geometry training
        sample_matrix = self.training_matrix if self.split == 'train' else _find_test_sample(self.training_matrix)
        print(f'{sample_matrix.sum()} of images are loading...')

        return sample_matrix


    def load_img(self):
        rays_savePath = Path(args.datadir) / f"rays_scaleType{args.rgbScaleType}_factor4green{args.factor4green}_idgeo{args.colIdx4RGBTrain}_\
            ndc{args.ndc_ray}_{self.split}_ds{self.downsample}_mtx{os.path.split(args.sample_matrix_dir)[1][:-4]}.pth"
        folders = [Path(args.datadir) / args.img_dir_name.replace('??', str(i)) 
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
            tensor_cropper = T.CenterCrop(args.crop_hw)
            tensor_resizer = T.Resize([H, W], antialias=True)
            for r, row in enumerate(sample_matrix):
                image_paths = sorted(glob.glob(str(folders[r] / f"images/*{args.img_ext}")))
                for c, aimEle in enumerate(row):
                    if aimEle != 1:
                        continue
                    elif c == args.colIdx4RGBTrain:
                        # it's for geometry training
                        ids4shapeTrain.append(len(all_rays))

                    image_path = image_paths[c]
                    c2w = torch.FloatTensor(self.poses[r])

                    img = self.read_non_raw(image_path)   # c h w [0-1]
                    if args.lsc:
                        img = LLFFDataset.img_correction(img)   # lens shade correction & black level correction
                    img = tensor_cropper(img)   # c croph cropw [0-1]
                    if self.downsample != 1.0:
                        img = tensor_resizer(img)

                    img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                    if args.factor4green != 1:
                        img[:, 1] = args.factor4green * img[:, 1]
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
            self.all_rgbs = torch.stack(all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
            self.all_poses = torch.stack(all_poses, 0)
            self.all_filtersIdx = torch.stack(all_filtersIdx, 0)

        if args.rgbScaleType == 'MAXRGB':
            # scale the rgb value
            maxrgb = self.all_rgbs.max()
            print('max rgb is :', maxrgb)
            self.all_rgbs /= maxrgb


    def load_colmap_depth(self, poses, bds_raw, basedir, factor=8, bd_factor=.75):
        croph, cropw = args.crop_hw
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
            # if id_im - 1 not in image_pick4depth:
            #     continue
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
        return data_list


    def get_depth_rays(self):
        W, H = self.img_wh

        data_list = []
        for ind, img_d in enumerate(self.depth_list):
            coord = torch.from_numpy(img_d['coord'])
            rays_o, rays_d = get_rays_by_coord_np(H, W, self.focal, self.poses[ind], coord)
            if args.ndc_ray == 1:
                rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            rayso_d = torch.cat([rays_o.float(), rays_d.float()], 1)  # (h*w, 6)

            data_list.append(rayso_d)
        return data_list


    def combine_depthimages(self):
        num = len(self.depth_list)
        depth_rays = []
        depth_weight = []
        depth_value = []

        for i in range(num):
            d_list = self.depth_list[i]
            d_rays = self.depth_rays[i]

            depth_rays.append(d_rays)
            depth_weight.append(torch.from_numpy(d_list['weight'][:, np.newaxis]))
            depth_value.append(torch.from_numpy(d_list['depth'][:, np.newaxis]))

        self.depth_rays = torch.cat(depth_rays, 0)
        self.depth_weight = torch.cat(depth_weight, 0)
        self.depth_value = torch.cat(depth_value, 0)

        LLFFDataset.depth_mean = self.depth_value.mean()

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

    @classmethod
    def img_correction(cls, img):
        return torch.clamp_max(torch.clamp_min(img - 0.014, 0.002) / cls.white, 1)
        # return (img - 0.014) / cls.white

    def transform(self, img, maxbits_num=65535.):
        if args.rgbScaleType == 'MAXBIT':
            return torch.FloatTensor(img / maxbits_num).permute(2, 0, 1)
        else:
            return torch.FloatTensor(np.asarray(img, dtype=np.float32)).permute(2, 0, 1)


def get_dataset4RGBtraining(dataset: LLFFDataset):
    allrays, allrgbs, allposesID, all_filterID = [], [], [], []
    id_in_group = dataset.ids4shapeTrain
    group_length = dataset.raysnum_oneimage

    for i in id_in_group:
        slc = slice(i * group_length, (i + 1) * group_length)

        allrays.append(dataset.all_rays[slc])
        allrgbs.append(dataset.all_rgbs[slc])
        allposesID.append(dataset.all_poses[slc])
        all_filterID.append(dataset.all_filtersIdx[slc])

    return torch.cat(allrays), torch.cat(allrgbs), torch.cat(allposesID), torch.cat(all_filterID)
