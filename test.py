# import numpy as np

# dirs = np.random.random([4, 3])
# c2w = np.random.random([3, 3])

# rays_d1 = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
# rays_d2 = dirs @ c2w[:3, :3].T

# pass

import cv2
import rawpy
from rawpy._rawpy import ColorSpace
import scipy.io as sio
import os
import numpy as np
white = sio.loadmat('./myspecdata/decorner/meanwhite.mat')['data'] ** (1 / 2.4)
black = sio.loadmat('./myspecdata/decorner/meanblack.mat')['data']
folder = r'myspecdata\filter20_no1\holiday\filter3img\images'
files = os.scandir(folder)


aimfolder = r'myspecdata\filter20_no1\holiday\images'
if not os.path.exists(aimfolder):
    os.mkdir(aimfolder)

for f in files:
    with rawpy.imread(f.path) as raw:
        rgb = raw.postprocess(output_color=ColorSpace.sRGB) / 255.0
        rgb = np.minimum(np.maximum(rgb - 0.014, 0.0) / white[..., np.newaxis], 1)

        rgb *= 255.0
        rgb = rgb.astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        aimfile = f'{aimfolder}\\{f.name[:-4]}.jpg'
        cv2.imwrite(aimfile, rgb)

        print('saved ', aimfile)
