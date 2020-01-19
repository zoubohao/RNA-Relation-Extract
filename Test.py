from nltk.stem.wordnet import  WordNetLemmatizer
import re
import nltk
from nltk.corpus import wordnet
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
# print(re.match(r'[0-9]',"fasdfasf"))
#
# def  is_chinese(word):
#     for c in word:
#         if u'\u4e00' <= c <= u'\u9fff':
#             return True
#     return False
#
#
# print(is_chinese("mir-26a-揳ssociated"))
# print(re.fullmatch(r'[0-9]*',"9.58"))
# print(WordNetLemmatizer().lemmatize("seems","n"))
# print(WordNetLemmatizer().lemmatize("choices","v"))
#
# def get_wordnet_pos(treebank_tag):
#     if treebank_tag.startswith('J'):
#         return wordnet.ADJ
#     elif treebank_tag.startswith('V'):
#         return wordnet.VERB
#     elif treebank_tag.startswith('N'):
#         return wordnet.NOUN
#     elif treebank_tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return wordnet.ADV
# tokens = nltk.word_tokenize("upregulationof dnmt1 as a result of mir-152 downregulation has been observed during hcc development")
# posTags = nltk.pos_tag(tokens)
# print(posTags)
# for ele in posTags:
#     print(nltk.stem.WordNetLemmatizer().lemmatize(ele[0],get_wordnet_pos(ele[1])))
#
#
# print(wordnet.ADJ,wordnet.ADV)
# print(nltk.stem.WordNetLemmatizer().lemmatize("newly",wordnet.ADJ))

# with open("./Data/669000_Negative.txt","r",encoding="utf-8") as rh:
#     with open("./Data/133770_Negative.txt","w",encoding="utf-8") as wh:
#         k=0
#         for line in rh:
#             oneLine = line.strip()
#             if k < 133770 :
#                 wh.write(oneLine + "\n")
#             else:
#                 break
#             k += 1

# import torchvision
# import torch
# import torchvision.transforms as transforms
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
# print("########")
# print(trainset)

# import matplotlib.pyplot  as plt
# import numpy as np
#
# def simulate(n,p,alpha,beta):
#     results = [p]
#     temp = p
#     for i in range(n):
#         print(temp)
#         temp = alpha * temp + (1. - beta) * (1. - temp)
#
#         results.append(temp)
#     return results
#
# result = simulate(10000,0.3,0.45,0.54)
# plt.plot(np.array(range(len(result))),result)
# plt.show()

# sen_len_list = []
# with open("./Data/Processed_2_Negative.txt","r") as rh:
#     for line in rh:
#         oneLine = line.strip()
#         sen_len_list.append(len(oneLine.split("\t")))
# with open("./Data/Processed_2_Positive.txt","r") as rh:
#     for line in rh:
#         oneLine = line.strip()
#         sen_len_list.append(len(oneLine.split("\t")))
# import matplotlib.pyplot as plt
# plt.hist(sen_len_list)
# plt.show()
#
# import numpy as np
# def weightCalculation(number_of_samples_of_one_class,beta):
#     return np.divide(np.subtract(1., beta),
#                      np.subtract(1., np.power(beta,number_of_samples_of_one_class))) + 0.
# print(weightCalculation(12877,0.8))
# print(weightCalculation(133270,0.8))


# import numpy as np
# positive_file = "./Data/Processed_2_Positive.txt"
# negative_file = "./Data/Processed_2_Negative.txt"
# vocabulary_file = "./Data/Vocabulary.txt"
# weight_save_path = "./ModelWeight/"
# loadWeight = False
# max_sequence_length = 44
# display_step = 25
# batch_size = 11
# positive_samples_in_one_batch = 1
# embedding_size = 10
# hidden_size = 128
# num_labels = 2
# epoch = 6
# lr = 1e-4
# decay_rate = 0.1
# save_model_steps = 5000
# trainOrTest = "train"
#
#
# vocab_map = {}
# positive_samples = []
# negative_samples = []
#
# with open(vocabulary_file,"r") as rh:
#     k = 1
#     for line in rh:
#         vocab_map[line.strip().split("\t")[0]] = k
#         k += 1
# print("There are " + str(len(vocab_map)) + " words.")
# vocabulary_len = len(vocab_map)
# with open(positive_file,"r") as rh:
#     for line in rh:
#         oneLine = line.strip()
#         tokens = oneLine.split("\t")
#         thisSentence = []
#         for token in tokens:
#             if len(thisSentence) < max_sequence_length:
#                 thisSentence.append(vocab_map[token])
#             else:
#                 break
#         if len(thisSentence) < max_sequence_length:
#             paddingNumber = max_sequence_length - len(thisSentence)
#             thisSentence = thisSentence + [0 for _ in range(paddingNumber)]
#         positive_samples.append(thisSentence)
# positive_samples = np.array(positive_samples,dtype=np.long)
# print("The shape of positive samples is ",positive_samples.shape)
# print(positive_samples)
# with open(negative_file,"r") as rh:
#     for line in rh:
#         oneLine = line.strip()
#         tokens = oneLine.split("\t")
#         thisSentence = []
#         for token in tokens:
#             if len(thisSentence) < max_sequence_length:
#                 thisSentence.append(vocab_map[token])
#             else:
#                 break
#         if len(thisSentence) < max_sequence_length:
#             paddingNumber = max_sequence_length - len(thisSentence)
#             thisSentence = thisSentence + [0 for _ in range(paddingNumber)]
#         negative_samples.append(thisSentence)
# negative_samples = np.array(negative_samples,dtype=np.long)
# print("The shape of negative samples is ",negative_samples.shape)
# print(negative_samples)
#
#
# train_positive_samples = positive_samples[0:-500]
# train_positive_labels = [1 for i in range(len(train_positive_samples))]
# print(train_positive_samples.shape)
#
# test_positive_samples = positive_samples[-500:]
# test_positive_labels = [1 for i in range(len(test_positive_samples))]
# print(test_positive_samples.shape)
#
# train_negative_samples = negative_samples[0:-500]
# train_negative_labels = [0 for i in range(len(train_negative_samples))]
# print(train_negative_samples.shape)
#
# test_negative_samples = negative_samples[-500:]
# test_negative_labels = [0 for i in range(len(test_negative_samples))]
# print(test_negative_samples.shape)
#
# negative_sample_size = train_negative_samples.shape[0]
# positive_sample_size = train_positive_samples.shape[0]
#
#
# from C_Model import FactorizedEmbedding
# facEmb = FactorizedEmbedding(vocabulary_len + 1,embedding_size,hidden_size)
# facEmb.train()
# trainingTimesInOneEpoch = max(negative_sample_size,positive_sample_size) // (batch_size - positive_samples_in_one_batch) + 1
# def DataGenerator(sample_list,label_list):
#     while True:
#         for s, sample in enumerate(sample_list):
#             yield sample , label_list[s]
#
# import torch
# positive_generator = DataGenerator(train_positive_samples, train_positive_labels)
# negative_generator = DataGenerator(train_negative_samples, train_negative_labels)
# for t in range(negative_sample_size + 5):
#     oneBatchSamples = []
#     oneBatchLabels = []
#     oneNegative, oneNegativeL = negative_generator.__next__()
#     oneBatchSamples.append(oneNegative)
#     oneBatchLabels.append(oneNegativeL)
#         # else:
#         #     oneNegative, oneNegativeL = negative_generator.__next__()
#         #     oneBatchSamples.append(oneNegative)
#         #     oneBatchLabels.append(oneNegativeL)
#     # print(oneBatchSamples)
#     # print("#########")
#     # print(np.array(oneBatchSamples)[index])
#     # print("#########")
#     # print(oneBatchLabels)
#     # print(np.array(oneBatchLabels)[index])
#     batchSamples = torch.from_numpy(np.array(oneBatchSamples)).long()
#     batchLabels = torch.from_numpy(np.array(oneBatchLabels)).long()
#     print(batchSamples)
#     print(batchLabels)
#     print(facEmb(batchSamples))
#     print(t)
# import torch
# print(torch.chunk(torch.ones(size=[3]),3))

# import numpy as np
# ones = np.ones(shape=[3,10],dtype=np.float32)
# zeros = np.zeros(shape=[5,10],dtype=np.float32)
# concat = np.concatenate([ones,zeros],axis=0)
# index = [i for i in range(8)]
# np.random.shuffle(index)
# print(concat)
# print(concat[index])
# print(index)

# import math
# lr = 1e-5
# for i in range(6):
#     lr = lr * math.pow(0.97, i / 1 + 0.)
#     print(lr)
# import torch
# max_len = 15
# d_model = 120
# pe = torch.zeros(max_len, d_model)
# position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
# div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
# pe[:, 0::2] = torch.sin(position * div_term)
# pe[:, 1::2] = torch.cos(position * div_term)
# pe = pe.unsqueeze(0).transpose(0, 1)
# print(pe.shape)
# print(pe[:12,:].shape)
# test = torch.zeros([12,8,120])
# print((test + pe[:12,:]).shape)


# import numpy as np
# import matplotlib.pyplot as plt
# import random
# from sklearn.gaussian_process import GaussianProcessRegressor
# a=np.random.random(50).reshape(50,1)
# b=a*2+np.random.random(50).reshape(50,1)
# plt.scatter(a,b,marker = 'o', color = 'r', label='3', s = 15)
# plt.show()
# gaussian=GaussianProcessRegressor()
# fiting=gaussian.fit(a,b)
# c=np.linspace(0.1,1,100)
# d=gaussian.predict(c.reshape(100,1))
# plt.scatter(a,b,marker = 'o', color = 'r', label='3', s = 15)
# plt.plot(c,d)
# plt.show()

# with open("./Data/669000_Negative.txt","r",encoding="utf-8") as rh:
#     k = 0
#     fileHandle = 1
#     wh = open("./Data/Paralle/" + str(fileHandle) + ".txt","w",encoding="utf-8")
#     for line in rh:
#         if k >= 133770:
#             signal = (k - 133770) // 107046
#             if signal != fileHandle - 1:
#                 print("CHNAGE")
#                 wh.flush()
#                 fileHandle = signal + 1
#                 wh.close()
#                 wh = open("./Data/Paralle/" + str(fileHandle) + ".txt", "w",encoding="utf-8")
#             print(k)
#             print(signal)
#             print(fileHandle-1)
#             wh.write(line)
#         k += 1
import torch
for _ in range(100):
    randInt = torch.randint(low=0, high=2, size=[1], dtype=torch.float32)
    if randInt == 1.:
        print("ok")
    else:
        print("no")


print(torch.zeros([2,3,4]) * torch.ones([2,3,4]))
torch.device("cuda")


