import torch
import numpy as np

def L2_Normalize(x):

    L2_Number = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    L2_Number = torch.unsqueeze(L2_Number, 1)
    one = torch.ones([1, x.shape[1]]).type(x.type())
    l2_matrix = torch.mm(L2_Number, one)
    return x / l2_matrix
