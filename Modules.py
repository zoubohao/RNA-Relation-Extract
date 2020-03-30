import torch
import torch.nn as nn
import math


class Attention(nn.Module):

    def __init__(self,dk,drop_p):
        super(Attention,self).__init__()
        self.dk = math.sqrt(dk)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, Q, K, V):
        """
        :param Q: A 3d tensor with shape of [N, T_q, d_model].
        :param K: A 3d tensor with shape of [N, T_k, d_model].
        :param V: A 3d tensor with shape of [N, T_k, d_model].
        :return:
        """
        QKt = torch.matmul(Q,torch.transpose(K,dim0=-2,dim1=-1)) / self.dk  # (N, T_q, T_k)
        output = torch.softmax(QKt,dim=-1)
        output = self.dropout(output)
        output = torch.matmul(output,V) # (N, T_q, d_v)
        return output

class Heads(nn.Module):

    def __init__(self,h,d_model,dk,drop_p):
        super(Heads,self).__init__()
        self.h = h
        self.dk = dk
        self.d_model = d_model
        self.QLinear = nn.Linear(d_model,d_model)
        self.KLinear = nn.Linear(d_model,d_model)
        self.VLinear = nn.Linear(d_model,d_model)
        self.attention = Attention(dk,drop_p)

    def forward(self, Q, K , V):
        bs = Q.size(0)
        qL = self.QLinear(Q).view([bs,-1,self.h,self.dk]).transpose(1,2)
        kL = self.KLinear(K).view([bs,-1,self.h,self.dk]).transpose(1,2)
        vL = self.VLinear(V).view([bs,-1,self.h,self.dk]).transpose(1,2)
        scores = self.attention(qL,kL,vL)
        catTensor = scores.transpose(1,2).contiguous().view([bs,-1,self.d_model])
        return catTensor


class MultiHeadAttention(nn.Module):

    def __init__(self,h = 8, d_model = 64 * 8,drop_p = 0.1):
        super(MultiHeadAttention,self).__init__()
        dk = d_model // h
        self.outLinear = nn.Linear(d_model , d_model)
        self.heads = Heads(h,d_model,dk,drop_p)

    def forward(self,x):
        q = x
        k = torch.clone(x)
        v = torch.clone(x)
        head = self.heads(q,k,v)
        return self.outLinear(head)

class FeedForward(nn.Module):

    def __init__(self,d_model,drop_p = 0.1):
        super(FeedForward,self).__init__()
        dff = d_model * 4
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
        self.sub1Dropout = nn.Dropout(drop_p)
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
        sub1F = self.sub1Ffw(x.clone())
        sub1Ln = self.lnsub1(sub1F)
        sub1Act = self.actSub1(sub1Ln)
        sub1Out = torch.mul(self.p1,self.sub1Dropout(sub1Act))
        ### sub 2
        sub2Lin = self.sub2Linear(x.clone())
        sub2Ln = self.lnsub2(sub2Lin)
        sub2Act = self.actSub2_1(sub2Ln)
        sub2Out = torch.mul(self.p2,self.sub2Dropout(sub2Act))
        return out2 + sub1Out + sub2Out




if __name__ == "__main__":
    testInput = torch.randn(size=[5,10,64*8])
    testMul = TransformerBlock()
    print(testMul)
    print(testMul(testInput).shape)









