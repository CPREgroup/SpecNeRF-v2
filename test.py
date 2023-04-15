import numpy as np

dirs = np.random.random([4, 3])
c2w = np.random.random([3, 3])

rays_d1 = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
rays_d2 = dirs @ c2w[:3, :3].T

pass
