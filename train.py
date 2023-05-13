import sys
sys.path.append('../myutils/')
sys.path.append('E:\pythonProject\python3\myutils_v2')

import os
from tqdm.auto import tqdm
from opt import config_parser, args
from dataLoader.IDsampler import get_simple_sampler
from shutil import copy
import scipy.io as sio
import json, random
from renderer import *
from utils import *
from tensorboardX import SummaryWriter
import datetime
from dataLoader.llff import LLFFDataset, get_dataset4RGBtraining
from dataLoader import dataset_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

lastfilename = 'null'
def saveModel(model, filepath):
    global lastfilename
    try:
        model.save(filepath)
    except Exception as e:
        print(traceback.format_exc())
        lastfilename = filepath
        return

    # code runs to here, only when the saving was successful
    # clear last 10 round model
    if os.path.exists(lastfilename):
        os.remove(lastfilename)
    lastfilename = filepath

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

    def set_batch(self, batch):
        self.batch = batch

@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)


@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

def reconstruction(args):

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset: LLFFDataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset: LLFFDataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh

    
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/SSFs', exist_ok=True)
    # copy config file
    copy(args.config, logfolder)

    summary_writer = SummaryWriter(logfolder)



    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))


    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(aabb, reso_cur, device,
                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far,
                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, distance_scale=args.distance_scale,
                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)

    tensorf: TensorVMSplit = tensorf.cuda()

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))
    criterian = torch.nn.MSELoss()
    specfix = SpectralFix()

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    if args.rgb4shape_endIter:
        allrays_4shape, allrgbs_4shape, allposesID_4shape, all_filterID_4shape = get_dataset4RGBtraining(train_dataset)
        rgb4shapeSampler = SimpleSampler(allrays_4shape.shape[0], args.batch_size - args.depth_supervise * args.depth_batchsize_endIter[0])

    allrays, allrgbs, allposesID, all_filterID = train_dataset.all_rays, train_dataset.all_rgbs, train_dataset.all_poses, train_dataset.all_filtersIdx
    if not args.ndc_ray:
        allrays, allrgbs, mask_filtered = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
        allposesID, all_filterID = allposesID[mask_filtered], all_filterID[mask_filtered]
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size - args.depth_supervise * args.depth_batchsize_endIter[0])

    if args.depth_supervise:
        depthrays, depthweights, depthvalue = \
            train_dataset.depth_rays, train_dataset.depth_weight, train_dataset.depth_value
        depthSampler = SimpleSampler(depthrays.shape[0], args.depth_batchsize_endIter[0])

    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")


    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        if iteration < args.rgb4shape_endIter:
            ray_idx = rgb4shapeSampler.nextids()
            rays_train, rgb_train, poseID_train, filterID_train = allrays_4shape[ray_idx], allrgbs_4shape[ray_idx].to(device), allposesID_4shape[ray_idx], all_filterID_4shape[ray_idx]
        else:
            ray_idx = trainingSampler.nextids()
            rays_train, rgb_train, poseID_train, filterID_train = allrays[ray_idx], allrgbs[ray_idx].to(device), allposesID[ray_idx], all_filterID[ray_idx]

        if args.reset_para and args.rgb4shape_endIter == iteration:
            #reset lr
            for param_group in optimizer.param_groups:
                if hasattr(param_group, 'myname'):
                    if param_group['myname'] in ['appLine', 'appPlane']:
                        param_group['lr'] = args.lr_init * 1.2
                    else:
                        param_group['lr'] = args.lr_basis * 1.2

        if args.depth_supervise:
            depth_rays_idx = depthSampler.nextids()
            depth_rays_train, depth_wei_train, depth_val_train = \
                depthrays[depth_rays_idx], depthweights[depth_rays_idx].to(device), depthvalue[depth_rays_idx].to(device)

            fake_poseid = torch.LongTensor([[0]]).expand((depth_rays_idx.shape[0], -1))
            fake_filterID = fake_poseid
            rays_train = torch.cat([rays_train, depth_rays_train])
            poseID_train = torch.cat([poseID_train, fake_poseid])
            filterID_train = torch.cat([filterID_train, fake_filterID])

        #rgb_map, alphas_map, depth_map, weights, uncertainty
        rgb_map, alphas_map, depth_map, weights, uncertainty, dist_loss, spec_map, ssf = \
            renderer(rays_train, tensorf, N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, \
                     is_train=True, poseids=poseID_train, filterids=filterID_train)

        if args.depth_supervise:
            rgb_batch = args.batch_size - args.depth_batchsize_endIter[0]
            # disentangle
            rgb_map = rgb_map[:rgb_batch]
            depth_map, depth_supervise = depth_map[:rgb_batch], depth_map[rgb_batch:]

            # depth map and loss
            depth_est_mapped = tensorf.depth_linear(depth_supervise)
            depth_loss = torch.mean((torch.abs(depth_val_train - depth_est_mapped)) * depth_wei_train)
            depth_loss_print = depth_loss.detach().item()
            summary_writer.add_scalar('train/depth_loss', depth_loss_print, global_step=iteration)
            # depth supervise only used for several rounds
            if iteration + 1 == args.depth_batchsize_endIter[1]:
                args.depth_supervise = False
                trainingSampler.set_batch(args.batch_size)
        else:
            depth_loss = depth_loss_print = 0

        if args.distortion_loss:
            dist_loss = 0.1 * dist_loss
            summary_writer.add_scalar('train/dist_loss', dist_loss, global_step=iteration)


        rgbloss = (((rgb_map - rgb_train) / (rgb_map.detach() + 0.01)) ** 2).mean()
        psnrloss = criterian(rgb_map, rgb_train).detach().item() # temp


        # loss
        total_loss = rgbloss + depth_loss + dist_loss

        if args.TV_weight_spec > 0:
            loss_specTV = TVloss_Spectral(spec_map)
            total_loss += loss_specTV * args.TV_weight_spec
            summary_writer.add_scalar('train/specTV', loss_specTV.detach().item(), global_step=iteration)

            # specfix_loss = specfix(spec_map)
            # total_loss += specfix_loss * 10
            # summary_writer.add_scalar('train/specFIX', specfix_loss.detach().item(), global_step=iteration)
        else:
            loss_specTV = 0

        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        PSNRs.append(-10.0 * np.log(psnrloss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', rgbloss, global_step=iteration)


        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            # print('\ndepth linear para (a, b) is', tensorf.depth_linear.a, tensorf.depth_linear.b, '\n')

            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {rgbloss:.6f}'
                + f' depth_loss = {depth_loss_print:.6f}'
                + f' dist_loss = {dist_loss:.6f}'
                + f' loss_specTV = {loss_specTV:.6f}'
            )
            PSNRs = []


        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            saveModel(tensorf, f'{logfolder}/{iteration}_{args.expname}.pth')
            sio.savemat(f'{logfolder}/SSFs/ssf_{iteration}.mat',
                        {'ssf': ssf.cpu().detach().numpy()})

            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)



        if iteration in update_AlphaMask_list:

            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)


            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays, allrgbs, mask_filtered = tensorf.filtering_rays(allrays,allrgbs)
                allposesID, all_filterID = allposesID[mask_filtered], all_filterID[mask_filtered]
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size - args.depth_supervise * args.depth_batchsize_endIter[0])


        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        

    tensorf.save(f'{logfolder}/{args.expname}.th')


    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    print(args)

    if args.export_mesh:
        export_mesh(args)
        exit(0)

    if args.render_only and (args.render_test or args.render_path):
        render_test(args)
    else:
        reconstruction(args)

