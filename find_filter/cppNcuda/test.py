import time
import numpy as np
import numpy_test as nt
import gascore

# bd = 20000

# a = np.ones([bd, bd], dtype=np.float32) * 2
# b = np.ones_like(a) * 2

# s1 = time.time()
# c = nt.test(a, b)
# print(time.time() - s1)
# print(c)

# s2 = time.time()
# d = nt.test_cuda(a, b)
# print(time.time() - s2)
# print(d)


a = [np.random.random([10, 2]) for _ in range(20)]

gascore.cal_score(a)

