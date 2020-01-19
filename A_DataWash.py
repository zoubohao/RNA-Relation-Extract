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
            word = "<RNA_Name>"
        elif word in protein:
            word = "<Protein_Name>"
        elif re.fullmatch(r'[\.0-9]*',word) is not None :
            word = "<Number>"
        if is_chinese(word) is True:
            word = ""
        word = WordNetLemmatizer().lemmatize(word,pos=get_wordnet_pos(pos))
        thisTokens.append(word)
    return thisTokens


if __name__ == "__main__" :

    inputPositiveFile = "./Data/13377_Positive.txt"
    inputNegativeFile = "./Data/133770_Negative.txt"
    RNANameFile = "./Data/RNA_Name.txt"
    proteinNameFile = "./Data/Protein_Name.txt"

    outputPositiveFile = "./Data/Processed_Positive.txt"
    outputNegativeFile = "./Data/Processed_Negative.txt"
    outputVocabularyFile = "./Data/Vocabulary_Frequency.txt"
    dropThreshold = 5

    rnaNameList = []
    proteinNameList = []

    print("RNA name read.")
    with open(RNANameFile,"r",encoding="utf-8") as rh:
        for line in rh:
            oneLine = line.strip()
            rnaNameList.append(oneLine)

    print("Protein name read.")
    with open(proteinNameFile,"r",encoding="utf-8") as rh:
        for line in rh:
            oneLine = line.strip()
            proteinNameList.append(oneLine)

    #wordsFrequencyMap = {}

    print("Input positive.")
    k = 0
    with open(inputPositiveFile,"r",encoding="utf-8") as rh:
        with open(outputPositiveFile,"a") as wh:
            for line in rh:
                oneLine = line.strip().split("\t")[1]
                tokens = Normalization(oneLine, rna=rnaNameList, protein=proteinNameList)
                if k % 50 == 0:
                    print(k)
                    print(oneLine)
                    print(tokens)
                k += 1
                for w in tokens:
                    wh.write(w+"\t")
                wh.write("\n")

    print("Completed.")

    print("Input Negative.")
    k = 0
    with open(inputNegativeFile,"r",encoding="utf-8") as rh:
        with open(outputNegativeFile,"a") as wh:
            for line in rh:
                oneLine = line.strip().split("\t")[1]
                tokens = Normalization(oneLine, rna=rnaNameList, protein=proteinNameList)
                if k % 50 == 0:
                    print(k)
                    print(oneLine)
                    print(tokens)
                k += 1
                for w in tokens:
                    wh.write(w + "\t")
                wh.write("\n")
    print("Completed.")

    # print("Write positive.")
    # with open(outputPositiveFile,"w",encoding="utf-8") as wh:
    #     for sen in positiveSen:
    #         for w in sen:
    #             if wordsFrequencyMap[w] >= dropThreshold:
    #                 wh.write(w + "\t")
    #             else:
    #                 wh.write("<UNK"+ str(np.random.randint(0,2)) +">" + "\t")
    #         wh.write("\n")
    # print("Completed.")
    #
    # print("Write Negative.")
    # with open(outputNegativeFile,"w",encoding="utf-8") as wh:
    #     for sen in negativeSen:
    #         for w in sen:
    #             if wordsFrequencyMap[w] >= dropThreshold:
    #                 wh.write(w + "\t")
    #             else:
    #                 wh.write("<UNK"+ str(np.random.randint(0,2)) +">" + "\t")
    #         wh.write("\n")
    # print("Completed.")
    #
    # with open(outputVocabularyFile,"w",encoding="utf-8") as wh:
    #     for key,value in wordsFrequencyMap.items():
    #         wh.write(key + "\t" + str(value) + "\n")






