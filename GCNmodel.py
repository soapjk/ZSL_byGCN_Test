import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models.resnet as resnet
import torchvision

class L2_normalization(nn.Module):
    def __init__(self,epsilon=1e-12):
        super(L2_normalization, self).__init__()

    def forward(self, x):

        L2_Number = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
        L2_Number = torch.unsqueeze(L2_Number, 1)
        one = torch.ones([1, x.shape[1]]).type(x.type())
        l2_matrix = torch.mm(L2_Number, one)
        return x / l2_matrix


class GraphConvolution(nn.Linear):
    def __init__(self, in_features, out_features,graph):
        super(GraphConvolution, self).__init__(in_features, out_features)
        self.A_hat = nn.Parameter(torch.from_numpy(graph), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        pass
    def forward(self, input):
        out = self.A_hat.mm(input).mm(self.weight)
        #out = out.matmul(self.weight)
        return out


class Mymodel(nn.Module):
    def __init__(self, graph):
        super(Mymodel, self).__init__()

        self.MLP_layers = nn.Sequential(
            GraphConvolution(300, 300, graph),
            nn.LeakyReLU(),
            GraphConvolution(300, 128, graph),
            #nn.LeakyReLU(),
            #GraphConvolution(256, 128, graph),
            #nn.LeakyReLU(),
            #GraphConvolution(128, 128, graph),
            L2_normalization()
        )

        def init_weights(m):
            if isinstance(m, nn.Linear):
                #nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.weight, -0.001, 0.001)
                nn.init.constant_(m.bias, 0)

        self.MLP_layers.apply(init_weights)

    def forward(self, semantic_mat):

        semantic_feature = self.MLP_layers(semantic_mat)

        return semantic_feature
