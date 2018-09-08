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
            GraphConvolution(300, 256, graph),
            nn.LeakyReLU(),
            GraphConvolution(256, 128, graph),
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


class GCN_JK(nn.Module):
    def __init__(self, graph):
        super(GCN_JK, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.uniform_(m.weight, -0.001, 0.001)
                nn.init.constant_(m.bias, 0)

        self.layer1 = nn.Sequential(
            GraphConvolution(300, 512, graph),
            nn.LeakyReLU()
        )
        """
        self.layer2 = nn.Sequential(
            GraphConvolution(2048, 2048, graph),
            nn.LeakyReLU()
        )
        """
        self.layer3 = nn.Sequential(
            GraphConvolution(512, 256, graph),
            nn.LeakyReLU()
        )
        """
        self.layer4 = nn.Sequential(
            GraphConvolution(1024, 1024, graph),
            nn.LeakyReLU()
        )
        """
        self.layer5 = nn.Sequential(
            GraphConvolution(256, 128, graph),
            nn.LeakyReLU()
        )
        self.layer6 = nn.Sequential(
            GraphConvolution(128, 128, graph),
            nn.LeakyReLU()
        )

        #self.weight = nn.Parameter(torch.Tensor(4, 1))

        self.fc = nn.Sequential(
            nn.Linear(512+256+128+128, 128),
            L2_normalization()
        )

        self.L2_nornal_layer = L2_normalization()
        self.layer1.apply(init_weights)
        #self.layer2.apply(init_weights)
        self.layer3.apply(init_weights)
        #self.layer4.apply(init_weights)
        self.layer5.apply(init_weights)
        self.layer6.apply(init_weights)
        self.fc.apply(init_weights)
        #nn.init.uniform_(self.weight, 0, 1)

    def forward(self, semantic_mat):
        x1 = self.layer1(semantic_mat)
        #x2 = self.layer2(x1)
        x3 = self.layer3(x1)
        #x4 = self.layer4(x3)
        x5 = self.layer5(x3)
        x6 = self.layer6(x5)
        concat = torch.cat((x1, x3, x5, x6), 1)

        #using_weight = torch.mm(self.weight, torch.ones(1, 128).type(self.weight.type())).view(1, -1)
        #filter_matrix = torch.cat((torch.eye(128), torch.eye(128), torch.eye(128), torch.eye(128)),0)
        #using_weight_matrix = torch.mm(torch.t(using_weight), torch.ones(1, 128).type(using_weight.type()))
        #using_weight_matrix=torch.mul(using_weight_matrix, filter_matrix.type(using_weight.type()))
        #semantic_feature = torch.mm(concat, using_weight_matrix)
        semantic_feature = self.fc(concat)
        semantic_feature=self.L2_nornal_layer(semantic_feature)
        return semantic_feature
