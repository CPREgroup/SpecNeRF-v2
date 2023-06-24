import random
import numpy as np
import scipy.io as sio
import filesort_int


def log(i, a, b):
    print("epoch --> ",
          str(i + 1).rjust(5, " "), " max:",
          str(round(a, 4)).rjust(8, " "), "mean:",
          str(round(b, 4)).rjust(8, " "), "alpha:",
          str(round(a / b, 4)).rjust(8, " "))


class GeneSolve:
## 初始定义，后续引用GeneSolve类方法为（初始种群数，最大迭代数，交叉概率，变异概率，最大适应度/平均适应度(扰动率趋于平稳则越接近1越好））
    def __init__(self, pose_path, ssf_path, pop_size, epoch, cross_prob, mutate_prob, alpha, poses, aim_poses, print_batch=10):
        self.aim_poses = aim_poses
        self.ssf_path = ssf_path
        self.pose_path = pose_path
        self.aim_number = aim_poses[0] * aim_poses[1]
        self.pop_size = pop_size
        self.epoch = epoch
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.print_batch = print_batch
        self.alpha = alpha
        self.poses = poses
        self.width = poses[0] * poses[1]
        self.best = None
        self.coor = self.prepare_poses()  # np N x 3
        self.fai = self.prepare_ssfs()  # 12 x 20 x 31
        self.dis_mtx = self.cal_distance() # np N x N

        # 产生初始种群
        genes = []
        for _ in range(self.pop_size):
            tts = []
            for _ in range(self.poses[0]):
                tt = ['0']*(self.poses[1]-self.aim_poses[1]) + ['1']*self.aim_poses[1]
                np.random.shuffle(tt)
                tts.append(tt)
            genes.append(np.array(tts))
        self.genes = np.array(genes)

    def prepare_ssfs(self):
        fs = filesort_int.sort_file_int(self.ssf_path, 'mat')
        filters_list = np.array([
            np.asarray(np.diagonal(sio.loadmat(x)['filter']), dtype=np.float32) for x in fs
        ])
        fai = filters_list[np.newaxis, ...].repeat(self.poses[0], axis=0)

        return fai

    def prepare_poses(self):
        coor = []
        poses_bounds = np.load(self.pose_path)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        for i in range(poses.shape[0]):
            coor.append(poses[i, :, 3])

        return coor

    def cal_distance(self):
        n = len(self.coor)
        dis_mtx = np.zeros((n, n))
        cal_dis = lambda a, b: np.sqrt(np.sum((self.coor[a]-self.coor[b]) ** 2))
        for r in range(n):
            for c in range(r+1, n):
                dis_mtx[r, c] = cal_dis(r, c)

        dis_mtx += dis_mtx.T
        return dis_mtx


    def inter_cross(self):
        """对染色体进行交叉操作"""
        ready_index = list(range(self.pop_size))
        while len(ready_index) >= 2:
            d1 = random.choice(ready_index)
            ready_index.remove(d1)
            d2 = random.choice(ready_index)
            ready_index.remove(d2)
            if np.random.uniform(0, 1) <= self.cross_prob:
                loc = random.choice(range(1, self.poses[1] - 1))
                temp = self.genes[d1][:, loc:]
                self.genes[d1][:, loc:] = self.genes[d2][:, loc:]
                self.genes[d2][:, loc:] = temp


    def mutate(self):
        """基因突变"""
        ready_index = list(range(self.pop_size))
        for i in ready_index:
            t0 = self.genes[i]
            if np.random.uniform(0, 1) <= self.mutate_prob:
                loc = random.choice(range(0, self.width))
                r, c = loc // self.poses[1], loc % self.poses[1]
                t0[r, c] = str(1 - int(t0[r, c]))
            # fix the total number to **
            for rr in range(self.poses[0]):
                t0[rr, :] = self.fix_total_number(list(t0[rr, :]))
            self.genes[i] = t0


    def fix_total_number(self, genes):
        ones_num = genes.count('1')
        genes = np.array(genes)
        if ones_num != self.aim_poses[1]:
            to_what = '1' if ones_num < self.aim_poses[1] else '0'

            tt = list(np.argwhere(genes == '1').squeeze())
            ones_idx = random.sample(tt, abs(ones_num - self.aim_poses[1]))
            genes[ones_idx] = to_what

        return genes


    def get_adjust(self):
        """计算适应度(只有在计算适应度的时候要反函数，其余过程全都是随机的二进制编码）"""
        x = self.get_decode()
        return x * np.sin(x) + 12

    def get_decode(self):
        """编码，从表现型到基因型的映射"""
        aimssfs = [
            self.fai[np.where(idx == '0', False, True)] for idx in self.genes
        ]

        return aimssfs


    def cycle_select(self):
        """通过轮盘赌来进行选择"""
        adjusts = self.get_adjust()
        if self.best is None or np.max(adjusts) > self.best[1]:
            self.best = self.genes[np.argmax(adjusts)], np.max(adjusts)
        p = adjusts / np.sum(adjusts)
        cu_p = []
        for i in range(self.pop_size):
            cu_p.append(np.sum(p[0:i]))
        cu_p = np.array(cu_p)
        r0 = np.random.uniform(0, 1, self.pop_size)
        sel = [max(list(np.where(r > cu_p)[0]) + [0]) for r in r0]
        # 保留最优的个体
        if np.max(adjusts[sel]) < self.best[1]:
            self.genes[sel[np.argmin(adjusts[sel])]] = self.best[0]
        self.genes = self.genes[sel]

    def evolve(self):
        """逐代演化"""
        for i in range(self.epoch):
            self.cycle_select()  #种群选取
            self.inter_cross()   #染色体交叉
            self.mutate()        #计算适应度
            a, b = np.max(self.get_adjust()), np.mean(self.get_adjust())
            if i % self.print_batch == self.print_batch - 1 or i == 0:
                log(i, a, b)
            if a / b < self.alpha:
                log(i, a, b)
                print("进化终止，算法已收敛！共进化 ", i + 1, " 代！")
                break


if __name__ == '__main__':
    gs = GeneSolve('../myspecdata/filter20_no1/newDoraemon/poses_bounds.npy',
                   '../myspecdata/filter20_no1/filters/*.mat',
                   100, 500, 0.65, 0.1, 1.2, (12, 20), (10, 3))
    gs.evolve()

