from C_Model import ALBERT
import torch
import torch.nn as nn
import math
import numpy as np
import sklearn.metrics as metrics


# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

positive_file = "./Data/Processed_2_Positive.txt"
negative_file = "./Data/Processed_2_Negative.txt"
vocabulary_file = "./Data/Vocabulary.txt"
weight_save_path = "./ModelWeight/"
loadWeight = False
trainModelLoad = 0.952
max_sequence_length = 42
display_step = 25
batch_size = 8
embedding_size = 10
hidden_size = 128
num_labels = 2
epoch = 2
lr = 1e-4
decay_rate = 0.1
decay_step = 18260
save_model_steps = 2025
trainOrTest = "train"
testModelSelect = 0.952


vocab_map = {}
positive_samples = []
negative_samples = []

with open(vocabulary_file,"r") as rh:
    k = 1
    for line in rh:
        vocab_map[line.strip().split("\t")[0]] = k
        k += 1
print("There are " + str(len(vocab_map)) + " words.")
vocabulary_len = len(vocab_map)
with open(positive_file,"r") as rh:
    for line in rh:
        oneLine = line.strip()
        tokens = oneLine.split("\t")
        thisSentence = []
        for token in tokens:
            if len(thisSentence) < max_sequence_length:
                thisSentence.append(vocab_map[token])
            else:
                break
        if len(thisSentence) < max_sequence_length:
            paddingNumber = max_sequence_length - len(thisSentence)
            thisSentence = thisSentence + [0 for _ in range(paddingNumber)]
        positive_samples.append(thisSentence)
positive_samples = np.array(positive_samples,dtype=np.long)
print("The shape of positive samples is ",positive_samples.shape)
print(positive_samples)
with open(negative_file,"r") as rh:
    for line in rh:
        oneLine = line.strip()
        tokens = oneLine.split("\t")
        thisSentence = []
        for token in tokens:
            if len(thisSentence) < max_sequence_length:
                thisSentence.append(vocab_map[token])
            else:
                break
        if len(thisSentence) < max_sequence_length:
            paddingNumber = max_sequence_length - len(thisSentence)
            thisSentence = thisSentence + [0 for _ in range(paddingNumber)]
        negative_samples.append(thisSentence)
negative_samples = np.array(negative_samples,dtype=np.long)
print("The shape of negative samples is ",negative_samples.shape)
print(negative_samples)


train_positive_samples = positive_samples[0:-500]
train_positive_labels = [1 for i in range(len(train_positive_samples))]
print(train_positive_samples.shape)

test_positive_samples = positive_samples[-500:]
print(test_positive_samples.shape)

train_negative_samples = negative_samples[0:-500]
train_negative_labels = [0 for i in range(len(train_negative_samples))]
print(train_negative_samples.shape)

test_negative_samples = negative_samples[-500:]
print(test_negative_samples.shape)

negative_sample_size = train_negative_samples.shape[0]
positive_sample_size = train_positive_samples.shape[0]

def DataGenerator(sample_list,label_list):
    while True:
        for s, sampleT in enumerate(sample_list):
            yield sampleT , label_list[s]

device = torch.device("cuda")
model = ALBERT(vocab_size=vocabulary_len,embed_size=embedding_size,hidden_size=hidden_size,
               num_heads=8,sequence_len=max_sequence_length,encoder_layers=10,num_encoder=8,num_labels=num_labels).to(device)
# for m in model.modules():
#     if isinstance(m , nn.Linear):
#         nn.init.kaiming_normal_(m.weight)
#         nn.init.constant_(m.weight, 0.)
print(model)
# weight=torch.from_numpy(np.array([positive_sample_size / negative_sample_size + 0.15, 1.])).float()
# 0.0966234
# 5 / 11
lossCri = nn.CrossEntropyLoss(reduction = "mean",weight=torch.from_numpy(np.array([1.5 / 6.5 - 0.005,1.0])).float()).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = lr,weight_decay=5e-3,nesterov=True,momentum=0.9)
#optimizer = torch.optim.SGD()

index = [i for i in range(batch_size)]
if trainOrTest.lower() == "train":
    positive_generator = DataGenerator(train_positive_samples, train_positive_labels)
    negative_generator = DataGenerator(train_negative_samples, train_negative_labels)
    if loadWeight :
        model.load_state_dict(torch.load(weight_save_path + "ALBERT_" + str(trainModelLoad) + ".pth"))
    model.train()
    trainingTimes = 0
    trainingTimesInOneEpoch = (negative_sample_size+positive_sample_size) // batch_size + 1
    print(trainingTimesInOneEpoch)
    print(decay_step)
    for e in range(epoch):
        for t in range(trainingTimesInOneEpoch):
            oneBatchSamples = []
            oneBatchLabels = []
            positive_samples_in_one_batch = np.random.randint(1,batch_size // 2 - 1)
            for b in range(batch_size):
                if b < positive_samples_in_one_batch:
                    onePositive, onePositiveL = positive_generator.__next__()
                    oneBatchSamples.append(onePositive)
                    oneBatchLabels.append(onePositiveL)
                else:
                    oneNegative, oneNegativeL = negative_generator.__next__()
                    oneBatchSamples.append(oneNegative)
                    oneBatchLabels.append(oneNegativeL)
            np.random.shuffle(index)
            batchSamples = torch.from_numpy(np.array(oneBatchSamples)[index]).long().to(device)
            batchLabels = torch.from_numpy(np.array(oneBatchLabels)[index]).long().to(device)
            optimizer.zero_grad()
            predictTensor = model(batchSamples)
            loss = lossCri(predictTensor, batchLabels)
            loss.backward()
            optimizer.step()
            if trainingTimes % display_step == 0:
                print("Predict tensor is ", predictTensor)
                print("Labels are ", batchLabels)
                print("Learning rate is ", optimizer.state_dict()['param_groups'][0]["lr"])
                print("Loss is ", loss)
                print("Training time is ", trainingTimes)
            if trainingTimes % decay_step == 0 and trainingTimes != 0:
                lr = lr * math.pow(decay_rate, trainingTimes / decay_step + 0.)
                state_dic = optimizer.state_dict()
                state_dic["param_groups"][0]["lr"] = lr
                optimizer.load_state_dict(state_dic)
            trainingTimes += 1
            if trainingTimes % save_model_steps == 0:
                torch.save(model.state_dict(), weight_save_path + "ALBERT_" + str(trainingTimes) + ".pth")
else:
    model.eval()
    model.load_state_dict(torch.load(weight_save_path + "ALBERT_" + str(testModelSelect) + ".pth"))
    predictLabels = []
    truthLabels = []
    print("POSITIVE SAMPLES PREDICT.")
    k = 0
    for sample in test_positive_samples:
        #print(torch.from_numpy(np.array([sample])).long().to(device).shape)
        predict = model(torch.from_numpy(np.array([sample])).long().to(device)).cpu().detach().numpy()
        position = np.argmax(np.squeeze(predict))
        print("##########")
        print(k)
        print(predict)
        print(position)
        predictLabels.append(position)
        truthLabels.append(1)
        k += 1
    for sample in test_negative_samples:
        #print(torch.from_numpy(np.array([sample])).long().to(device).shape)
        predict = model(torch.from_numpy(np.array([sample])).long().to(device)).cpu().detach().numpy()
        print("##########")
        print(k)
        print(predict)
        position = np.argmax(np.squeeze(predict))
        print(position)
        predictLabels.append(position)
        truthLabels.append(0)
        k += 1
    acc = metrics.accuracy_score(y_pred=predictLabels,y_true=truthLabels)
    print("Accuracy is : ",acc)
    confusionMatrix = metrics.confusion_matrix(y_true=truthLabels,y_pred=predictLabels)
    print("The confusion matrix is : ",confusionMatrix)
    precision = metrics.precision_score(y_true=truthLabels,y_pred=predictLabels)
    recall = metrics.recall_score(y_true=truthLabels,y_pred=predictLabels)
    print("Precision is : ",precision)
    print("Recall is : ",recall)
    f1_score = metrics.f1_score(y_true=truthLabels,y_pred=predictLabels)
    print("F1 score is ",f1_score)






