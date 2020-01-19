import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Block(nn.Module):

    def __init__(self,inChannels,outChannels):
        super(CNN_Block,self).__init__()
        self.cnn1 = nn.Conv2d(inChannels,outChannels,kernel_size=3,padding=1)

    def forward(self, x):
        return self.cnn1(x)

class StochasticBlock(nn.Module):

    def __init__(self,inChannels,outChannels):
        super(StochasticBlock,self).__init__()
        self.cnnBlock = CNN_Block(inChannels,outChannels)

    def forward(self, x):
        xShape = x.shape
        if self.training:
            randInt = torch.randint(low=0, high=2, size=xShape, dtype=torch.float32,requires_grad=False)
        else:
            randInt = 1
        mask = torch.zeros(xShape,requires_grad=False)
        if randInt == 1.:
            mask = torch.ones(xShape,requires_grad=False)
        cnnOutput = self.cnnBlock(x)
        finalTensor = torch.mul(mask,cnnOutput)
        return finalTensor + x
