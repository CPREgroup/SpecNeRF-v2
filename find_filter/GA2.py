import itertools
import os
import random
import sys
sys.path.append(r'E:\pythonProject\python3\myutils_v2')
sys.path.append(os.getcwd())
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
import scipy.io as sio
from tqdm.auto import tqdm

import filesort_int
from Online_breakpoint_debug import Online_breakpoint_debug
from multiprocessing.dummy import Pool
from scipy.spatial.transform import Rotation
from myutils import Plt_subplot_in_loop_or_ion
import gascore

myd = Online_breakpoint_debug()
myd.start()

def execute_time(func):
    # 定义嵌套函数，用来打印出装饰的函数的执行时间
    def wrapper(*args, **kwargs):
        # 定义开始时间
        start = time.time()
        # 执行函数
        func_return = func(*args, **kwargs)
        # 定义结束时间
        end = time.time()
        # 打印方法名称和其执行时间
        print('{}() execute time: {}s'.format(func.__name__, end-start))
        # 返回func的返回值
        return func_return

    # 返回嵌套的函数
    return wrapper

def normalization(a):
    ran = a.max() - a.min()
    a = (a - a.min()) / ran
    return a, ran


def log(pbar, i, a, b):
    d = "epoch --> " + \
        str(i + 1).rjust(5, " "), " max:" + \
        str(round(a, 4)).rjust(8, " "), "mean:" + \
        str(round(b, 4)).rjust(8, " "), "alpha:" + \
        str(round(a / b, 4)).rjust(8, " ")
    # pbar.set_description(str(d))
    print(d)


class GeneSolve:
    ## 初始定义，后续引用GeneSolve类方法为（初始种群数，最大迭代数，交叉概率，变异概率，最大适应度/平均适应度(扰动率趋于平稳则越接近1越好））
    def __init__(self, pose_path, ssf_path, pop_size, epoch, cross_prob, mutate_prob, alpha,
                 poses, aim_number, sigma, print_batch=2):

        assert pop_size % 2 == 0, 'pop size must be even number'

        self.sigma = sigma
        self.poses = poses
        self.ssf_path = ssf_path
        self.pose_path = pose_path
        self.aim_number = aim_number
        self.pop_size = pop_size
        self.epoch = epoch
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.print_batch = print_batch
        self.alpha = alpha
        self.width = poses[0] * poses[1]
        self.best = None
        self.coor, self.eulers, self.view_dirs = self.prepare_poses()  # list N x 3, 12 x 3, 12 x 3
        self.fai = self.prepare_ssfs()  # np 12 x 20 x 31
        self.ssf_coor = self.cal_ssfCoorelation()
        self.dis_mtx = self.cal_distance()  # np N x N
        self.cosSimi = self.cal_cos_simi()

        self._idxs_4_inter_cross = np.arange(0, stop=pop_size).astype(dtype=np.int)

        # sio.savemat('coordinate_ssf_distance.mat', {
        #     'coor': np.array(self.coor),
        #     'ssfs': self.fai,
        #     'distance': self.dis_mtx
        # })
        def cross2idx(pair, loc, genes):
            d1, d2 = pair
            d1_a, d1_b = genes[d1, 0:loc], genes[d1, loc:]
            d2_a, d2_b = genes[d2, 0:loc], genes[d2, loc:]
            genes[d1] = np.append(d1_a, d2_b)
            genes[d2] = np.append(d2_a, d1_b)

        self.crossfunc = np.vectorize(cross2idx, excluded=['genes'],
                                 signature='(j),()->()', cache=True)

        @nb.jit(nopython=True, parallel=True)
        def gen(pop_size, width):
            # 产生初始种群
            genes = []
            for _ in range(pop_size):
                tt = np.array([0] * (width - aim_number) + [1] * aim_number)
                np.random.shuffle(tt)
                genes.append(tt)
            return genes

        self.genes = np.vstack(gen(self.pop_size, self.width))

        def get_meshgrid_comb():
            n = number
            init_id = list(range(1, n))
            res_id = init_id[:]
            for i in range(1, n):
                res_id += list(map(lambda x: x + n*i, init_id[i:]))
            return res_id
        self.comb_meshgrid_id = get_meshgrid_comb()

        pass

    def cal_ssfCoorelation(self):
        n = self.poses[1]
        ssf_norm = np.linalg.norm(np.array(self.filters_list), axis=-1)

        @nb.njit(parallel=True)
        def cal(n, ssfs, ssf_norm):
            ssfCor_norm = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1):
                    # tt = np.dot(ssfs[i], ssfs[j]) / (ssf_norm[i] * ssf_norm[j])
                    ''' use var normed '''
                    tt = np.var(ssfs[i] / ssf_norm[i] + ssfs[j] / ssf_norm[j])
                    ssfCor_norm[i, j] = ssfCor_norm[j, i] = tt
            return ssfCor_norm

        return cal(n, self.filters_list, ssf_norm)

    def cal_cos_simi(self):
        num = self.view_dirs @ self.view_dirs.T
        normed = np.linalg.norm(self.view_dirs, axis=-1).reshape(-1, 1)
        all_normed = normed * normed.T

        cosSimi = (num / all_normed) / 2 + 0.5
        cosSimi = cosSimi ** cosSim_gamma

        return cosSimi

    def prepare_ssfs(self):
        fs = filesort_int.sort_file_int(self.ssf_path, 'mat')[1:]
        filters_list = np.array([
            np.asarray(np.diagonal(sio.loadmat(x)['filter']), dtype=np.float32) for x in fs
        ])
        fai = filters_list[np.newaxis, ...].repeat(self.poses[0], axis=0)

        self.filters_list = filters_list
        self.filters_trans_mean = filters_list.mean(axis=1)

        return fai

    def prepare_poses(self):
        coor = []
        view_dirs = []

        foo_v = np.array([0.3, 0.3, 0.7]).reshape([-1, 1])
        hypothetic_vect = foo_v / np.linalg.norm(foo_v)

        poses_bounds = np.load(self.pose_path)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        # Original poses has rotation in form "down right back", change to "right up back"
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

        for i in range(poses.shape[0]):
            coor.append(poses[i, :, 3])
            view_dirs.append(poses[i, :, :3] @ hypothetic_vect)

        # matrix to angles,
        r = Rotation.from_matrix(poses[..., :3])
        euler_angles = r.as_euler('zxy', True)

        return coor, euler_angles, np.array(view_dirs).squeeze()

    def cal_distance(self):
        n = len(self.coor)
        map_dis = lambda d: np.exp((-d ** 2) / self.sigma ** 2)

        @nb.njit(parallel=True)
        def cal(coor):
            cal_dis = lambda a, b: np.sqrt(np.sum((coor[a] - coor[b]) ** 2))
            dis_mtx = np.zeros((n, n))
            for r in range(n):
                for c in range(r + 1, n):
                    tt = cal_dis(r, c)
                    dis_mtx[r, c] = dis_mtx[c, r] = tt
            return dis_mtx

        dis_mtx = cal(self.coor)

        dis_mtx, _ = normalization(dis_mtx)
        dis_mtx = map_dis(6 * dis_mtx)

        return dis_mtx

    
    def inter_cross(self):
        np.random.shuffle(self._idxs_4_inter_cross)
        idxs = self._idxs_4_inter_cross.reshape([-1, 2])
        idxs2change = idxs[np.random.random(idxs.shape[0]) <= self.cross_prob]
        locs = np.random.randint(1, self.width-2, idxs2change.shape[0])

        self.crossfunc(pair=idxs2change, loc=locs, genes=self.genes)

    
    def mutate(self):
        """基因突变"""
        ready_index = list(range(self.pop_size))
        for i in ready_index:
            t0 = self.genes[i]
            if np.random.uniform(0, 1) <= self.mutate_prob:
                loc = random.choice(range(0, self.width))
                t0[loc] = 1 - t0[loc]
            # fix the total number to **
            self.genes[i] = self.fix_total_number(t0)

    def fix_total_number(self, genes):
        ones_num = genes.sum()
        if ones_num != self.aim_number:
            to_what = 1 if ones_num < self.aim_number else 0

            tt = list(np.argwhere(genes == (1 - to_what)).squeeze())
            try:
                ones_idx = random.sample(tt, abs(ones_num - self.aim_number))
            except Exception as e:
                print(traceback.format_exc())
                myd.goin(locals())
            genes[ones_idx] = to_what

        return genes

    
    def get_combination(self, arr):
        comb = np.array(np.meshgrid(arr, arr)).T.reshape(-1, 2)
        comb2 = comb[self.comb_meshgrid_id, :]

        return comb2


    def get_adjust(self):
        """编码，从表现型到基因型的映射"""
        indexes = np.argwhere(self.genes == 1)[:, 1]
        r_idx, c_idx = np.unravel_index(indexes, self.fai.shape[:2])
        r_idx, c_idx = r_idx.reshape([-1, number]), c_idx.reshape([-1, number])

        """计算适应度(只有在计算适应度的时候要反函数，其余过程全都是随机的二进制编码）"""
        r_comb = np.stack(map(self.get_combination, r_idx), axis=0).reshape(-1, 2) # e.g. (40000 x 780) x 2
        c_comb = np.stack(map(self.get_combination, c_idx), axis=0).reshape(-1, 2)

        distance = self.dis_mtx[r_comb[:, 0], r_comb[:, 1]]
        ssfCoorela = self.ssf_coor[c_comb[:, 0], c_comb[:, 1]]
        viewDir_rela = self.cosSimi[r_comb[:, 0], r_comb[:, 1]]
        score = (distance * ssfCoorela * viewDir_rela).reshape(self.pop_size, -1).sum(1)

        score = 1 / score

        return score

    
    def cycle_select(self):
        """通过轮盘赌来进行选择"""
        adjusts = self.get_adjust()
        if self.best is None or np.max(adjusts) >= self.best[1]:
            self.best = self.genes[np.argmax(adjusts)], np.max(adjusts)
        p = adjusts / np.sum(adjusts)

        cu_p = np.cumsum(p)

        r0 = np.random.uniform(0, 1, self.pop_size)
        wheel_choose_func = np.vectorize(np.searchsorted, [np.int], excluded=['a'], cache=True)
        sel = wheel_choose_func(a=cu_p, v=r0) #[wheel_choose(r, cu_p) for r in r0]

        # sel = [np.append(np.where(r > cu_p)[0], 0).max() for r in r0]
        # sel = list(map(lambda r:np.append(np.where(r > cu_p)[0], 0).max(), r0))
        # 保留最优的个体
        if np.max(adjusts[sel]) < self.best[1]:
            self.genes[sel[np.argmin(adjusts[sel])]] = self.best[0]
        self.genes = self.genes[sel]

    def evolve(self):
        def show_best(best):
            score, gene = best
            g = gene.reshape(self.poses)
            plt.imshow(g)
            plt.show()
            print(g.sum())

            sio.savemat(f'find_filter_res/GA_{self.aim_number}_{score}_{random.randint(0, 50)}.mat', {'mask': g})

        """逐代演化"""
        best = (0, None)
        pbar = tqdm(range(self.epoch), miniters=self.print_batch, file=sys.stdout)
        with Plt_subplot_in_loop_or_ion(1, 1) as myplt:
            for i in pbar:
                self.cycle_select()  # 种群选取
                self.inter_cross()  # 染色体交叉
                self.mutate()

                adjust_val = self.get_adjust()  # 计算适应度
                best_gene, a, b = self.genes[np.argmax(adjust_val)], np.max(adjust_val), np.mean(adjust_val)

                if i % self.print_batch == 0:
                    g = best_gene.reshape(self.poses)
                    myplt.repaint()
                    myplt.ax_cur().imshow(g)
                    myplt.pause()
                    # save a temp
                    sio.savemat(f'{outfile}/GA_{self.aim_number}_ep{i}_{a}.mat',
                                {'mask': g})

                if a >= best[0]:
                    best = (a, best_gene)

                if i % self.print_batch == self.print_batch - 1 or i == 0:
                    log(pbar, i, a, b)
                if a / b < self.alpha:
                    log(pbar, i, a, b)
                    print("进化终止，算法已收敛！共进化 ", i + 1, " 代！")
                    break

        show_best(best)


if __name__ == '__main__':
    outfile = f'find_filter/find_filter_res/xjhdesk_sigma0d5_wei0d1_num40' # lab_trans_sigma0d3_wei1d0_num40
    if not os.path.exists(outfile):
        os.makedirs(outfile)

    sigma = 0.5
    number = 40
    cosSim_gamma = 2.5
    weight = 0.1
    print(f'running sigma = {sigma}')
    print(f'running number = {number}')
    t1 = time.time()
    gs = GeneSolve(r'./myspecdata\filters19_optimized\xjhdesk/poses_bounds.npy',
                   r'./myspecdata\filters19_optimized\filters_interp25/*.mat',
                   40000, 2000, 0.65, 0.1, 1.05, (9, 19), number, np.sqrt(sigma))
    gs.evolve()
    t2 = time.time()

    print(f'cost {(t2 - t1) / 60.0} mins')
