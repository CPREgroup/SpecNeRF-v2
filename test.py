# import numpy as np

# dirs = np.random.random([4, 3])
# c2w = np.random.random([3, 3])

# rays_d1 = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
# rays_d2 = dirs @ c2w[:3, :3].T

# pass

# import cv2
# import rawpy
# from rawpy._rawpy import ColorSpace
# import scipy.io as sio
# import os
# import numpy as np
# white = sio.loadmat('./myspecdata/decorner/meanwhite.mat')['data'] ** (1 / 2.4)
# black = sio.loadmat('./myspecdata/decorner/meanblack.mat')['data']
# folder = r'myspecdata\filter20_no1\holiday\filter3img\images'
# files = os.scandir(folder)


# aimfolder = r'myspecdata\filter20_no1\holiday\images'
# if not os.path.exists(aimfolder):
#     os.mkdir(aimfolder)

# for f in files:
#     with rawpy.imread(f.path) as raw:
#         rgb = raw.postprocess(output_color=ColorSpace.sRGB) / 255.0
#         rgb = np.minimum(np.maximum(rgb - 0.014, 0.0) / white[..., np.newaxis], 1)

#         rgb *= 255.0
#         rgb = rgb.astype(np.uint8)
#         rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

#         aimfile = f'{aimfolder}\\{f.name[:-4]}.jpg'
#         cv2.imwrite(aimfile, rgb)

#         print('saved ', aimfile)


# import torch
# from torch_efficient_distloss import eff_distloss, eff_distloss_native, flatten_eff_distloss

# # A toy example
# B = 8192  # number of rays
# N = 128   # number of points on a ray
# w = torch.rand(B, N).cuda()
# w = w / w.sum(-1, keepdim=True)
# w = w.clone().requires_grad_()
# s = torch.linspace(0, 1, N+1).cuda()
# m = (s[1:] + s[:-1]) * 0.5
# m = m[None].repeat(B,1)
# interval = 1/N

# loss = 0.01 * eff_distloss(w, m, interval)
# loss.backward()
# print('Loss', loss)
# print('Gradient', w.grad)

# a = torch.FloatTensor([[1,2],[3,4]])
# print(torch.softmax(a, 1))
# print(a / a.sum(-1, keepdim=True))

import torch

# 构造矩阵A和列表B
A = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
B = [1, 3, 1, 2, 2]

# 使用索引操作构造矩阵C
C = A[B]

# 使用torch.stack函数将C的多行堆叠成一个张量
C = torch.stack(C, dim=0)

pass
