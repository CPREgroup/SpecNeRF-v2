import numpy as np
import scipy.io as sio

f = sio.loadmat('myspecdata/filters19_optimized/xjhdesk/exhibition/ssfs.mat')['ssfs'][0].tolist()

print(f)

