from C_Model import ALBERT
import torch
import numpy as np
import sklearn.metrics as metrics
import nltk
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

def  is_chinese(word):
    for c in word:
        if u'\u4e00' <= c <= u'\u9fff':
            return True
    return False

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.ADV

def Normalization(text,rna,protein):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\-\']'," ",text)
    words = nltk.word_tokenize(text)
    posTages = nltk.pos_tag(words)
    thisTokens = []
    for tage in posTages:
        word = tage[0]
        pos = tage[1]
        if word in rna:
            word = "<NC_RNA>"
        elif word in protein:
            word = "<mRNA>"
        elif re.fullmatch(r'[\.0-9]*',word) is not None :
            word = "<Number>"
        if is_chinese(word) is True:
            word = ""
        word = WordNetLemmatizer().lemmatize(word,pos=get_wordnet_pos(pos))
        thisTokens.append(word)
    return thisTokens

### "./Data/Paralle/" + str(i+1) + ".txt"
def testNegative(negativeFile):
    vocabulary_file = "./Data/Vocabulary.txt"
    model_weight = "./ModelWeight/ALBERT_0.952.pth"
    RNANameFile = "./Data/RNA_Name.txt"
    proteinNameFile = "./Data/Protein_Name.txt"

    rnaNameList = []
    proteinNameList = []
    print("RNA name read.")
    with open(RNANameFile, "r", encoding="utf-8") as rh:
        for line in rh:
            oneLine = line.strip()
            rnaNameList.append(oneLine)

    print("Protein name read.")
    with open(proteinNameFile, "r", encoding="utf-8") as rh:
        for line in rh:
            oneLine = line.strip()
            proteinNameList.append(oneLine)

    vocab_map = {}
    with open(vocabulary_file, "r") as rh:
        k = 1
        for line in rh:
            vocab_map[line.strip().split("\t")[0]] = k
            k += 1
    embedding_size = 10
    hidden_size = 128
    max_sequence_length = 42
    device = torch.device("cuda")
    model = ALBERT(vocab_size=len(vocab_map), embed_size=embedding_size, hidden_size=hidden_size,
                   num_heads=8, sequence_len=max_sequence_length, encoder_layers=10, num_encoder=8, num_labels=2).to(device)
    model.eval()
    model.load_state_dict(torch.load(model_weight))
    predictLabels = []
    truthLabels = []
    with open(negativeFile, "r", encoding="utf-8") as rh:
        k = 0
        for line in rh:
            oneLine = line.strip("\n").split("\t")[1]
            tokens = Normalization(oneLine, rnaNameList, proteinNameList)
            thisSentence = []
            for token in tokens:
                if len(thisSentence) < max_sequence_length:
                    if token in vocab_map:
                        thisSentence.append(vocab_map[token])
                    else:
                        thisSentence.append(vocab_map["<UNK>"])
                else:
                    break
            if len(thisSentence) < max_sequence_length:
                paddingNumber = max_sequence_length - len(thisSentence)
                thisSentence = thisSentence + [0 for _ in range(paddingNumber)]
            print("##########")
            print(k)
            print(tokens)
            # print(thisSentence)
            predict = model(torch.from_numpy(np.array([thisSentence])).long().to(device)).cpu().detach().numpy()
            position = np.argmax(np.squeeze(predict))
            print(predict)
            print(position)
            predictLabels.append(position)
            truthLabels.append(0)
            k += 1
    acc = metrics.accuracy_score(y_pred=predictLabels, y_true=truthLabels)
    print("Accuracy is : ", acc)


def testNegativeCPU(negativeFile):
    vocabulary_file = "./Data/Vocabulary.txt"
    model_weight = "./ModelWeight/ALBERT_0.952.pth"
    RNANameFile = "./Data/RNA_Name.txt"
    proteinNameFile = "./Data/Protein_Name.txt"

    rnaNameList = []
    proteinNameList = []
    print("RNA name read.")
    with open(RNANameFile, "r", encoding="utf-8") as rh:
        for line in rh:
            oneLine = line.strip()
            rnaNameList.append(oneLine)

    print("Protein name read.")
    with open(proteinNameFile, "r", encoding="utf-8") as rh:
        for line in rh:
            oneLine = line.strip()
            proteinNameList.append(oneLine)

    vocab_map = {}
    with open(vocabulary_file, "r") as rh:
        k = 1
        for line in rh:
            vocab_map[line.strip().split("\t")[0]] = k
            k += 1
    embedding_size = 10
    hidden_size = 128
    max_sequence_length = 42
    model = ALBERT(vocab_size=len(vocab_map), embed_size=embedding_size, hidden_size=hidden_size,
                   num_heads=8, sequence_len=max_sequence_length, encoder_layers=10, num_encoder=8, num_labels=2)
    model.eval()
    model.load_state_dict(torch.load(model_weight))
    predictLabels = []
    truthLabels = []
    with open(negativeFile, "r", encoding="utf-8") as rh:
        k = 0
        for line in rh:
            oneLine = line.strip("\n").split("\t")[1]
            tokens = Normalization(oneLine, rnaNameList, proteinNameList)
            thisSentence = []
            for token in tokens:
                if len(thisSentence) < max_sequence_length:
                    if token in vocab_map:
                        thisSentence.append(vocab_map[token])
                    else:
                        thisSentence.append(vocab_map["<UNK>"])
                else:
                    break
            if len(thisSentence) < max_sequence_length:
                paddingNumber = max_sequence_length - len(thisSentence)
                thisSentence = thisSentence + [0 for _ in range(paddingNumber)]
            print("##########")
            print(k)
            print(tokens)
            # print(thisSentence)
            predict = model(torch.from_numpy(np.array([thisSentence])).long()).detach().numpy()
            position = np.argmax(np.squeeze(predict))
            print(predict)
            print(position)
            predictLabels.append(position)
            truthLabels.append(0)
            k += 1
    acc = metrics.accuracy_score(y_pred=predictLabels, y_true=truthLabels)
    print("Accuracy is : ", acc)





