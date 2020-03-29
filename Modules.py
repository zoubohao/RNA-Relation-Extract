import torch
import torch.nn as nn
import math


class Attention(nn.Module):

    def __init__(self,dk,dropout_p):
        super(Attention,self).__init__()
        self.dk = math.sqrt(dk)
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, Q, K, V):
        """
        :param Q: A 3d tensor with shape of [N, T_q, d_model].
        :param K: A 3d tensor with shape of [N, T_k, d_model].
        :param V: A 3d tensor with shape of [N, T_k, d_model].
        :return:
        """
        QKt = torch.matmul(Q,torch.transpose(K,dim0=1,dim1=2)) / self.dk  # (N, T_q, T_k)
        output = torch.softmax(QKt,dim=-1)
        output = self.dropout(output)
        output = torch.matmul(output,V) # (N, T_q, d_v)
        return output

class Head(nn.Module):

    def __init__(self,d_model,dk,dv,drop_p):
        super(Head,self).__init__()
        self.QLinear = nn.Linear(d_model,dk)
        self.KLinear = nn.Linear(d_model,dk)
        self.VLinear = nn.Linear(d_model,dv)
        self.attention = Attention(dk,drop_p)

    def forward(self, Q, K , V):
        qProject = self.QLinear(Q)
        kProject = self.KLinear(K)
        vProject = self.VLinear(V)
        attention = self.attention(qProject,kProject,vProject)
        return attention

class MultiHeadAttention(nn.Module):

    def __init__(self,h = 8, d_model = 64 * 8,drop_p = 0.1):
        super(MultiHeadAttention,self).__init__()
        dk = dv = d_model // h
        self.outLinear = nn.Linear(h * dv , d_model)
        self.heads = nn.ModuleList([Head(d_model,dk,dv,drop_p) for _ in range(h)])


    def forward(self,x):
        q = x
        k = torch.clone(x)
        v = torch.clone(x)
        projectionList = []
        for oneHead in self.heads:
            projectionList.append(oneHead(q,k,v))
        catTensor = torch.cat(projectionList,dim=-1)
        return self.outLinear(catTensor)

class FeedForward(nn.Module):

    def __init__(self,d_model,drop_p = 0.1):
        super(FeedForward,self).__init__()
        dff = 2048
        self.linear1 = nn.Linear(d_model,dff)
        self.linear2 = nn.Linear(dff,d_model)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(dff)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x1 = self.linear1(x)
        x1 = self.ln(x1)
        x1 = self.activation(x1)
        x2 = self.linear2(x1)
        return self.dropout(x2)

class TransformerBlock(nn.Module):

    def __init__(self,h = 8,d_model = 512,drop_p = 0.1):
        super(TransformerBlock,self).__init__()
        self.multiAttention = MultiHeadAttention(h,d_model,drop_p)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.feedForward = FeedForward(d_model,drop_p)
        self.alphaA = nn.Parameter(torch.zeros(size=[1],requires_grad=True).float(),requires_grad=True)
        self.alphaB = nn.Parameter(torch.zeros(size=[1], requires_grad=True).float(), requires_grad=True)
        ### For enhancing the ability of non-linearity, i added another path to get the result.
        ### sub 1
        self.p1 = nn.Parameter(data=torch.tensor(data=0.), requires_grad=True)
        self.sub1Ffw= FeedForward(d_model,drop_p)
        self.lnsub1 = nn.LayerNorm(d_model)
        self.actSub1 = nn.GELU()
        ### sub 2
        self.p2 = nn.Parameter(data=torch.tensor(data=0.), requires_grad=True)
        self.sub2Dropout = nn.Dropout(drop_p)
        self.sub2Linear = nn.Linear(d_model,d_model)
        self.actSub2_1 = nn.GELU()
        self.lnsub2 = nn.LayerNorm(d_model)


    def forward(self, x):
        ### main
        xOri = x.clone()
        attention = self.multiAttention(x)
        ln1 = self.ln1(x + attention)
        out1 = xOri + torch.mul(ln1,self.alphaA)
        fft = self.feedForward(out1)
        ln2 = self.ln2(fft + out1)
        out2 = out1 + torch.mul(ln2,self.alphaB)
        ### sub 1
        sub1F = self.sub1Ffw(x)
        sub1Ln = self.lnsub1(sub1F)
        sub1Act = self.actSub1(sub1Ln)
        sub1Out = torch.mul(self.p1,sub1Act)
        ### sub 2
        sub2Drop = self.sub2Dropout(x)
        sub2Lin = self.sub2Linear(sub2Drop)
        sub2Ln = self.lnsub2(sub2Lin)
        sub2Act = self.actSub2_1(sub2Ln)
        sub2Out = torch.mul(self.p2,sub2Act)
        return out2 + sub1Out + sub2Out


class TransformerEncoder(nn.Module):

    def __init__(self,layerNumber = 3,h = 8,d_model = 512,drop_p = 0.1):
        super(TransformerEncoder,self).__init__()
        self.layerNumber = layerNumber
        self.transformers = nn.ModuleList([TransformerBlock(h,d_model,drop_p=drop_p) for _ in range(layerNumber)])

    def forward(self, x):
        feature = x.clone()
        for transformer in self.transformers:
            feature = transformer(feature)
        return feature


if __name__ == "__main__":
    testInput = torch.randn(size=[5,10,64*8])
    testMul = TransformerEncoder()
    print(testMul)
    print(testMul(testInput).shape)









