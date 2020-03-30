import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np



class MishActivation(nn.Module):

    def __init__(self):
        super(MishActivation,self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class FeedForward(nn.Module):

    def __init__(self,inFeatures,dropout):
        super(FeedForward,self).__init__()
        self.liner1 = nn.Linear(in_features=inFeatures,out_features=inFeatures * 4)
        self.liner2 = nn.Linear(in_features=inFeatures * 4,out_features=inFeatures)
        self.ln = nn.LayerNorm(inFeatures * 4)
        self.dropOut = nn.Dropout(dropout)
        self.act = MishActivation()
        self.act1 = MishActivation()

    def forward(self, x):
        liner1T = self.liner1(x)
        bnT = self.ln(liner1T)
        act0T = self.act(bnT)
        dropT = self.dropOut(act0T)
        liner2T = self.liner2(dropT)
        act1T = self.act1(liner2T)
        return act1T


class TransformerEncoder(nn.Module):

    def __init__(self,hidden_size,num_heads):
        super(TransformerEncoder,self).__init__()
        ##1
        self.multiA0 = nn.MultiheadAttention(hidden_size,num_heads=num_heads,dropout=0.2)
        self.mulLn = nn.LayerNorm(hidden_size)
        self.fft0 = FeedForward(hidden_size, dropout=0.2)
        self.ln0 = nn.LayerNorm(hidden_size)
        self.p1 = nn.Parameter(data=torch.tensor(data=0.), requires_grad=True)
        ##2
        self.fft1 = FeedForward(hidden_size, dropout=0.2)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.p2 = nn.Parameter(data=torch.tensor(data=0.), requires_grad=True)
        ##3
        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(hidden_size,hidden_size)
        self.act = MishActivation()
        self.ln2 = nn.LayerNorm(hidden_size)
        self.p3 = nn.Parameter(data=torch.tensor(data=0.), requires_grad=True)


    ###[S,N,E]
    def forward(self, x):
        """
        :param x: [batch size, sequence length, embed dim]
        :return: the output of encoder,shape is the same as x
        """
        stackPs = torch.stack([self.p1,self.p2,self.p3])
        softMaxPs = torch.softmax(stackPs,dim=0)
        p0 , p1 , p2 = torch.chunk(softMaxPs,3,dim=0)
        xCopy = x.clone()
        x1 = x.clone()
        x2 = x.clone()
        ### main block
        multiA0T , _ = self.multiA0(x.transpose(0,1) ,x.clone().transpose(0,1),x.clone().transpose(0,1),need_weights=False)
        mainAddT = multiA0T.transpose(0,1) + xCopy
        mainLn = self.mulLn(mainAddT)
        mainLnCopy = mainLn.clone()
        fft0T = self.fft0(mainLn)
        addFFT0T = fft0T + mainLnCopy
        ln0T = self.ln0(addFFT0T)
        ### sub 1 block
        x1Copy = x1.clone()
        fft1T = self.fft1(x1)
        ln1T = self.ln1(fft1T + x1Copy)
        ### simple sub 2 block
        x2Copy = x2.clone()
        fft2T = self.linear2(self.dropout2(x2))
        actT = self.act(fft2T)
        ln2T = self.ln2(actT + x2Copy)
        ### add to one tensor
        addedTensor = torch.mul(ln0T,p0) + torch.mul(ln1T,p1) + torch.mul(ln2T,p2)
        return addedTensor


class FactorizedEmbedding(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size):
        """
        :param vocab_size:
        :param embed_size:
        :param hidden_size: hidden_size must much larger than embed_size
        """
        super(FactorizedEmbedding,self).__init__()
        self.embeddingLayer = nn.Embedding(vocab_size + 1,embed_size,padding_idx=0,scale_grad_by_freq=True)
        self.liner = nn.Linear(embed_size,hidden_size)

    def forward(self, x):
        """
        :param x: [batch,sequences]
        :return: [batch,sequences,hidden_size]
        """
        embedTensor = self.embeddingLayer(x)
        linerTensor = self.liner(embedTensor)
        return linerTensor

def add_nTensors(tensorsList):
    addR = tensorsList[0] + tensorsList[1]
    for t in range(2,len(tensorsList)):
        addR = addR + tensorsList[t]
    return addR

import math
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
# This module is copied from Github.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.2).
        max_len: the max. length of the incoming sequence.
    """

    def __init__(self, d_model, max_len, dropout=0.2,):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ALBERT(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size,num_heads,sequence_len,encoder_layers,num_encoder,num_labels):
        super(ALBERT,self).__init__()
        self.num_labels = num_labels
        self.encoderLayers = encoder_layers
        self.numberOfEncoders = num_encoder
        self.embedding = FactorizedEmbedding(vocab_size,embed_size,hidden_size)
        self.positionalEmbedding = PositionalEncoding(hidden_size,max_len=sequence_len,dropout=0.2)
        self.encoders = nn.ModuleList([TransformerEncoder(hidden_size,num_heads) for _ in range(num_encoder)])
        self.linear = nn.Linear(sequence_len * hidden_size, num_labels)
        self.fbn = nn.BatchNorm1d(num_features=sequence_len * hidden_size)
        self.fact = MishActivation()

    def forward(self, x):
        tempResult = []
        embeddingResult = self.embedding(x)
        positionalEmbeddingT = self.positionalEmbedding(embeddingResult.transpose(0,1)).transpose(0,1)
        for e in range(self.numberOfEncoders):
            tempTensor = positionalEmbeddingT.clone()
            for _ in range(self.encoderLayers):
                tempTensor = self.encoders[e](tempTensor)
            tempResult.append(tempTensor)
        meanT = add_nTensors(tempResult)
        _ , s , h = meanT.shape
        flatten = torch.reshape(meanT,[-1,s*h])
        fBNT = self.fbn(flatten)
        outputTensor = torch.softmax(self.linear(self.fact(fBNT)),dim=-1)
        return outputTensor



if __name__ == "__main__":
    device = torch.device("cuda")
    testInput = torch.from_numpy(np.ones(shape=[4,10])).long().to(device)
    model = ALBERT(vocab_size=10,embed_size=5,hidden_size=128,num_heads=4,encoder_layers=8,num_encoder=3,num_labels=2,sequence_len=10).to(device)












