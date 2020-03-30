#from C_Model import ALBERT
import torch
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
from torch.optim.rmsprop import  RMSprop
from C_OtherModel import ALBERT
from LearningRateSch import CosineDecaySchedule

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

positive_file = "./Data/Processed_2_Positive.txt"
negative_file = "./Data/Processed_2_Negative.txt"
vocabulary_file = "./Data/Vocabulary.txt"
weight_save_path = "./ModelWeight/"
loadWeight = True
trainModelLoad = 3000 ### 3000
max_sequence_length = 42
display_step = 10
batch_size = 5
embedding_size = 10
#hidden_size = 128
hidden_size = 512
num_labels = 2
epoch = 2  ## 29230
lr = 1e-4
save_model_steps = 1000
trainOrTest = "train"
scheduleMinLR = 5e-6
scheduleMaxIniIter = 1600
scheduleDecayRate = 0.94
scheduleFactor = 1.234
### Test
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
    indexes = list(range(len(sample_list)))
    np.random.shuffle(indexes)
    while True:
        for s in indexes:
            yield sample_list[s] , label_list[s]

device = torch.device("cuda")
# model = ALBERT(vocab_size=vocabulary_len,embed_size=embedding_size,hidden_size=hidden_size,
#                num_heads=8,sequence_len=max_sequence_length,encoder_layers=10,num_encoder=8,num_labels=num_labels).to(device)
model = ALBERT(vocab_size=vocabulary_len,embed_size=embedding_size,d_model=hidden_size,num_labels=num_labels,sequence_len=max_sequence_length,
               drop_p=0.15,cross_layers=6,parallel_Transformers=7,total_layers=4).to(device)
for m in model.modules():
    if isinstance(m , nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.)
print(model)
# weight=torch.from_numpy(np.array([positive_sample_size / negative_sample_size + 0.15, 1.])).float()
# 0.0966234
# 5 / 11
lossCri = nn.CrossEntropyLoss(reduction = "sum",weight=torch.from_numpy(np.array([1.5 / 6.5,1.0])).float()).to(device)
optimizer = RMSprop(model.parameters(), lr = lr,weight_decay=5e-3,momentum=0.85,eps=1e-6,alpha=0.96)

index = [i for i in range(batch_size)]
if trainOrTest.lower() == "train":
    scheduler = CosineDecaySchedule(lrMin=scheduleMinLR, lrMax=lr, tMaxIni=scheduleMaxIniIter, factor=scheduleFactor,
                                    lrDecayRate=scheduleDecayRate)
    positive_generator = DataGenerator(train_positive_samples, train_positive_labels)
    negative_generator = DataGenerator(train_negative_samples, train_negative_labels)
    if loadWeight :
        model.load_state_dict(torch.load(weight_save_path + "ALBERT_" + str(trainModelLoad) + ".pth"))
    model.train()
    trainingTimes = 0
    trainingTimesInOneEpoch = (negative_sample_size+positive_sample_size) // batch_size + 1
    print(trainingTimesInOneEpoch)
    for e in range(epoch):
        for t in range(trainingTimesInOneEpoch):
            oneBatchSamples = []
            oneBatchLabels = []
            positive_samples_in_one_batch = np.random.randint(1,batch_size // 2)
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
            scheduler.step()
            if trainingTimes % display_step == 0:
                print("#################")
                print("Predict tensor is ", predictTensor)
                print("Labels are ", batchLabels)
                print("Learning rate is ", optimizer.state_dict()['param_groups'][0]["lr"])
                print("Loss is ", loss)
                print("Training time is ", trainingTimes)
            learning_rate = scheduler.calculateLearningRate()
            state_dic = optimizer.state_dict()
            state_dic["param_groups"][0]["lr"] = float(learning_rate)
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
    fpr, tpr, thresholds = metrics.roc_curve(y_true=truthLabels,y_score=predictLabels,pos_label=1)
    auc = metrics.auc(fpr,tpr)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of This ALBERT')
    plt.legend(loc="lower right")
    plt.show()






