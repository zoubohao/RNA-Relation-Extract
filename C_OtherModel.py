from Modules import TransformerBlock
import torch
import torch.nn as nn


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

    def __init__(self,vocab_size,embed_size,d_model,sequence_len,num_labels,
                 cross_layers = 8,parallel_Transformers = 8, total_layers = 10 ,drop_p = 0.15):
        super(ALBERT,self).__init__()
        self.cross_layers = cross_layers
        self.parallel_transformers = parallel_Transformers
        self.div = float(parallel_Transformers)
        self.total_layers = total_layers

        self.num_labels = num_labels
        self.embedding = FactorizedEmbedding(vocab_size,embed_size,d_model)
        self.positionalEmbedding = PositionalEncoding(d_model,max_len=sequence_len,dropout=0.2)
        encoders = {}
        for l in range(cross_layers):
            encoders["EncoderLayer_"+str(l)] = nn.ModuleList([TransformerBlock(d_model = d_model,drop_p = drop_p) for _ in range(parallel_Transformers)])
        self.encoders = nn.ModuleDict(encoders)
        self.linear = nn.Linear(sequence_len * d_model, num_labels)
        self.fact = nn.GELU()

    def forward(self, x):
        embeddingResult = self.embedding(x)
        positionalEmbeddingT = self.positionalEmbedding(embeddingResult.transpose(0,1)).transpose(0,1)
        inputTensor = positionalEmbeddingT.clone()
        for _ in range(self.total_layers):
            for c in range(self.cross_layers):
                tempTensors = []
                oneLayerEncoders = self.encoders["EncoderLayer_"+str(c)]
                for m in oneLayerEncoders:
                    outT = m(inputTensor)
                    tempTensors.append(outT)
                inputTensor = add_nTensors(tempTensors) / self.div
        encodedTensor = inputTensor.clone()
        _ , s , h = encodedTensor.shape
        flatten = encodedTensor.view([-1,s*h])
        outputTensor = self.linear(self.fact(flatten))
        return outputTensor



if __name__ == "__main__":
    import numpy as np
    testInput = torch.from_numpy(np.ones(shape=[4,10])).long()
    model = ALBERT(vocab_size=10,embed_size=5,d_model=512,num_labels=2,sequence_len=10,drop_p=0.1)
    result = model(testInput)
    print(result)
    print(result.shape)




