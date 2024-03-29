import math
import torch,os,imageio,sys
from tqdm.auto import tqdm
from dataLoader.ray_utils import get_rays
from models.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
from utils import *
from dataLoader.ray_utils import ndc_rays_blender
from dataLoader.llff import LLFFDataset
from opt import args
import scipy.io as sio

def OctreeRender_trilinear_fast(rays, tensorf, chunk=args.chunk_size, N_samples=-1, ndc_ray=False, white_bg=True, \
                                is_train=False, device='cuda', **kargs):
    poseids, filterids = kargs['poseids'], kargs['filterids']
    filters = LLFFDataset.filters_back

    rgbs, alphas, depth_maps, weights, uncertainties, dist_losses, spec_maps = [], [], [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        poseids_chunk = poseids[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        filters_chunk = filters[filterids[chunk_idx * chunk:(chunk_idx + 1) * chunk].reshape(-1)].to(device)
    
        rgb_map, depth_map, dist_loss, spec_map, phi = tensorf(rays_chunk, poseids_chunk, filters_chunk, is_train=is_train, white_bg=white_bg, \
                                                ndc_ray=ndc_ray, N_samples=N_samples)

        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
        dist_losses.append(dist_loss)
        spec_maps.append(spec_map)
    
    return None if args.render_test_exhibition else torch.cat(rgbs), None, torch.cat(depth_maps), None, None, \
          torch.cat(dist_losses).mean(), torch.cat(spec_maps), phi

@torch.no_grad()
def evaluation(test_dataset:LLFFDataset,tensorf, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/spec", exist_ok=True)
    # delete old spec.mat
    del_list = os.listdir(f'{savePath}/spec')
    for f in del_list:
        file_path = os.path.join(f'{savePath}/spec', f)
        if os.path.isfile(file_path):
            os.remove(file_path)


    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _, _, spec_map, _ = \
            renderer(rays, tensorf, N_samples=N_samples, ndc_ray=ndc_ray, white_bg = white_bg, device=device, \
                     poseids=test_dataset.all_poses[idxs[idx]], filterids=test_dataset.all_filtersIdx[idxs[idx]])
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map, spec_map = rgb_map.reshape(H, W, args.observation_channel).cpu(), depth_map.reshape(H, W).cpu(), spec_map.reshape(H, W, args.spec_channel).cpu().numpy()

        depth_map_raw = depth_map.numpy().copy()
        if rgb_map.shape[-1] != 1:
            depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        else:
            depth_map = visualize_depth_numpy_mono(depth_map.numpy(),near_far)

        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, args.observation_channel)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', tensorf.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', tensorf.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            # imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
            # save spec.mat
            sio.savemat(f'{savePath}/spec/{prtx}{idx:03d}.mat', {'spec': spec_map})
            sio.savemat(f'{savePath}/rgbd/{prtx}{idx:03d}_depth.mat', {'depth': depth_map_raw})

    # imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    # imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs

@torch.no_grad()
def evaluation_path(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):
    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/spec", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    ones_filtersIdx = torch.LongTensor([[0]])
    for idx, c2w in tqdm(enumerate(c2ws)):

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        rgb_map, _, depth_map, _, _, _, spec_map, _ = \
            renderer(rays, tensorf, N_samples=N_samples, ndc_ray=ndc_ray, white_bg = white_bg, device=device, \
                     poseids=ones_filtersIdx.expand((rays.shape[0], -1)), filterids=ones_filtersIdx.expand((rays.shape[0], -1)))
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map, spec_map = rgb_map.reshape(H, W, args.observation_channel).cpu(), depth_map.reshape(H, W).cpu(), spec_map.reshape(H, W, args.spec_channel).cpu().numpy()

        if rgb_map.shape[-1] != 1:
            depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        else:
            depth_map = visualize_depth_numpy_mono(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)
            sio.savemat(f'{savePath}/spec/{prtx}{idx:03d}.mat', {'spec': spec_map})


    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs


@torch.no_grad()
def exhibition(test_dataset,tensorf, c2ws, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
                    white_bg=False, ndc_ray=False, device='cuda', scale=False, **kargs):
    rgb_maps, depth_maps = [], []
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)
    os.makedirs(savePath+"/spec", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    ones_filtersIdx = torch.LongTensor([[0]])
    filtersets = kargs['filtersets']
    ssfs = kargs['ssfs']
    lights = kargs['lights']
    try:
        lightOrigin = torch.from_numpy(sio.loadmat(args.exhibition_lightorigin_path)['light_spec'][:, args.band_start_idx:]).cuda()
        lightOrigin = (1 / lightOrigin.mean()) * lightOrigin # normalize to one mean
    except Exception as e:
        print('Just warning you, seems you did not provide lightorgin file.')
    for idx, c2w in tqdm(enumerate(c2ws)):
        times1 = math.ceil(len(c2ws) / len(filtersets))
        if idx % times1 == 0:
            filter = torch.from_numpy(filtersets[idx // times1]).cuda()
            
        times2 = math.ceil(len(c2ws) / len(ssfs))
        if idx % times2 == 0:
            ssf = torch.from_numpy(ssfs[idx // times2]).cuda()

        if len(lights) != 0:
            if idx % math.ceil(len(c2ws) / len(lights)) == 0:
                light = torch.from_numpy(lights[idx // math.ceil(len(c2ws) / len(lights))]).cuda()
        else:
            light = None

        W, H = test_dataset.img_wh

        c2w = torch.FloatTensor(c2w)
        rays_o, rays_d = get_rays(test_dataset.directions, c2w)  # both (h*w, 3)
        if ndc_ray:
            rays_o, rays_d = ndc_rays_blender(H, W, test_dataset.focal[0], 1.0, rays_o, rays_d)
        rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)

        _, _, depth_map, _, _, _, spec_map, _ = \
            renderer(rays, tensorf, N_samples=N_samples, ndc_ray=ndc_ray, white_bg = white_bg, device=device, \
                     poseids=ones_filtersIdx.expand((rays.shape[0], -1)), filterids=ones_filtersIdx.expand((rays.shape[0], -1)))
        
        if light is not None:
            spec_map = spec_map * light / lightOrigin
        rgb_map = ((spec_map * filter) @ ssf)
        if scale:
            rgb_map = (0.6 / torch.quantile(rgb_map.reshape(-1), 0.95)) * rgb_map
        rgb_map = rgb_map.clamp(0,1)

        rgb_map, depth_map, spec_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu(), spec_map.reshape(H, W, args.spec_channel).cpu().numpy()

        if rgb_map.shape[-1] != 1:
            depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        else:
            depth_map = visualize_depth_numpy_mono(depth_map.numpy(),near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=8)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=8)
