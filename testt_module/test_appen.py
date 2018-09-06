import sys
sys.path.append('../')
import torch
import numpy as np
from SimpleClassfier.Classfier import Classfier
model = torch.load('../SimpleClassfier/classfier_model/classfier.pt')
model.cpu()
weight = model.weight.detach().numpy()
np.save('../train_data/classfierweight.npy', weight)
pass
