

wordFrequencyMap = {}

inputPositiveFile = "./Data/Processed_Positive.txt"
inputNegativeFile = "./Data/Processed_Negative.txt"
outputPositiveFile = "./Data/Processed_2_Positive.txt"
outputNegativeFile = "./Data/Processed_2_Negative.txt"
outputVocabularyFile = "./Data/Vocabulary.txt"
dropThreshold = 5

positiveSentences = []
negativeSentences = []

print("positive")
with open(inputPositiveFile,"r") as rh:
    for line in rh:
        oneLine = line.strip()
        tokens = oneLine.split("\t")
        positiveSentences.append(tokens)
        for w in tokens:
            if w not in wordFrequencyMap:
                wordFrequencyMap[w] = 1
            else:
                freq = wordFrequencyMap[w]
                freq += 1
                wordFrequencyMap[w] = freq
print("negative")
with open(inputNegativeFile,"r") as rh:
    for line in rh:
        oneLine = line.strip()
        tokens = oneLine.split("\t")
        negativeSentences.append(tokens)
        for w in tokens:
            if w not in wordFrequencyMap:
                wordFrequencyMap[w] = 1
            else:
                freq = wordFrequencyMap[w]
                freq += 1
                wordFrequencyMap[w] = freq
print("writing positive")
with open(outputPositiveFile,"w") as wh:
    for tokens in positiveSentences:
        for w in tokens:
            if wordFrequencyMap[w] >= dropThreshold:
                wh.write(w + "\t")
            else:
                wh.write("<UNK>" + "\t")
        wh.write("\n")
print("writing negative")
with open(outputNegativeFile,"w") as wh:
    for tokens in negativeSentences:
        for w in tokens:
            if wordFrequencyMap[w] >= dropThreshold:
                wh.write(w + "\t")
            else:
                wh.write("<UNK>" + "\t")
        wh.write("\n")
print("writing vocabulary")
with open(outputVocabularyFile,"w") as wh:
    wh.write("<UNK>" + "\t" + "0" + "\n")
    for key,value in wordFrequencyMap.items():
        if value >= dropThreshold:
            wh.write(key + "\t" + str(value) + "\n")




