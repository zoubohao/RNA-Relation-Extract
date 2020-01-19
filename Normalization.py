import torch as t
import torch.nn as nn


class BatchNorm(nn.Module):

    def __init__(self,imageSize,momentum,eps = 1e-5):
        """
        :param momentum: The momentum of running mean
        :param imageSize: [C,H,W]
        """
        super(BatchNorm,self).__init__()
        self.momentum = momentum
        self.weight = nn.Parameter(t.tensor(1.,dtype=t.float32),True)
        self.bias = nn.Parameter(t.tensor(0., dtype=t.float32), True)
        self.running_mean = t.zeros(imageSize, dtype=t.float32,requires_grad=False)
        self.running_variance= t.zeros(imageSize, dtype=t.float32, requires_grad=False)
        self.eps = eps


    def forward(self, x):
        """
        :param x: The shape of x is [b,c,h,w], we do the mean of b dimension.
        :return:
        """
        ### update running mean and running variance
        if self.training :
            sample_mean = t.mean(x, dim=0)
            sample_variance = t.var(x, dim=0,unbiased=False)
            self.running_mean = self.running_mean * self.momentum + (1. - self.momentum) * sample_mean
            self.running_variance = self.running_variance * self.momentum + (1. - self.momentum) * sample_variance
            center_x = (x - sample_mean) / t.sqrt(sample_variance + self.eps)
            shift = t.mul(self.weight,center_x) + self.bias
            return shift
        else:
            center_x = (x - self.running_mean) /  t.sqrt(self.running_variance + self.eps)
            return t.mul(self.weight,center_x) + self.bias


class 




if __name__ == "__main__":
    import numpy as np
    testBN = BatchNorm([3,3,3],momentum=0.9)
    testInputTensor = t.from_numpy(np.arange(0,100,0.1)[0:-1]).view([-1,3,3,3]).float()
    testBN.eval()
    print(testInputTensor.shape)
    result = testBN(testInputTensor)
    print(result)
    for pa in testBN.parameters():
        print(pa)