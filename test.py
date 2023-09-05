import numpy as np
from scipy.spatial.distance import pdist, squareform

import torch

def calculate_sum_diff(arr):
    tensor_arr = torch.tensor(arr)
    diff_matrix = tensor_arr.view(-1, 1) - tensor_arr
    upper_triangle = torch.triu(diff_matrix, diagonal=1).abs()
    result = torch.sum(upper_triangle)
    return result

array = [5, 2, 3]  # 用实际的值代替 a, b, c, d
result = calculate_sum_diff(array)
print(result)
