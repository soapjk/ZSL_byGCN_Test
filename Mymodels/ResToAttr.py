import sys
sys.path.append('../')

import math
import torchvision.models.resnet as resnet
import torch
import torch.nn as nn
from Data_ready import MyDataset
from torch.utils.data import DataLoader

class ResToAttr(nn.Module):

    def __init__(self):
        self.out_size = 30
        layers = [2, 2, 2, 2]
        block = resnet.BasicBlock
        self.inplanes = 64
        super(ResToAttr, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=1)
        #self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(512 * block.expansion, 30)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        """
        l2_number = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
        l2_number = torch.unsqueeze(l2_number, 1)
        one_vector = torch.ones([1,128]).float().cuda()
        maxtrix_l2 = torch.mm(l2_number, one_vector)
        x = x / maxtrix_l2
        """
        return x

    def get_loss(self, x, label):
        x = self.forward(x)
        loss = nn.MSELoss()(x, label)
        return loss


    def predict(self, image, attribute):
        with torch.no_grad():
            self.eval()
            n_class = attribute.size(0)
            batch_size = attribute.size(0)
            visual_emb = self.forward(image)

            visual_emb = visual_emb.repeat(1, n_class).view(batch_size * n_class, -1)
            #semantic_emb = self.word_emb_transformer(attribute).repeat(batch_size, 1)

            score = (visual_emb - attribute).norm(dim=1).view(batch_size, n_class)
            #unseen_mask = np.load('../Data/data/eval_data/eval_unseen_mask.npy')
            #unseen_mask = torch.from_numpy(unseen_mask).repeat(batch_size,1).float().cuda()
            #score = score + unseen_mask
            pred = score.min(1)[1]

        return pred.detach().cpu().numpy()

