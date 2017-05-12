import numpy as np
import re
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras.layers.core import Activation
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import load_model

f = open("../data/normaliseOLiATaggedTrain.tsv").readlines()
f_test = open("../data/johannes.olia.tag.tsv").readlines()


pos = []

def createData(f):
    #Reads the input data , outputs the training data in an array: [[word, annotation],...]
    data = []
    sentence = []
    for line in f:
        split_line = re.split(" |\t",line.strip("\n"))
        sentence.append(split_line)
        if  not (line and line.strip()):
            data.append(sentence[:-2])
            sentence = []

    all_x = []
    train_x = []
    for s in data:
        for d in s:
            annotation = '|'.join(sorted(d[1:]))
            train_x.append([d[0],annotation])
        all_x.append(train_x)
        train_x = []
        short_x = [x for x in all_x if len(x) < 64]
        X = [[c[0] for c in x] for x in short_x]
        y = [[c[1] for c in y] for y in short_x]
    return (all_x,X,y)


all_x,X,y = createData(f)
lengths = [len(x) for x in all_x]
print 'Input sequence length range: ', max(lengths), min(lengths)
all_text = [c for x in X for c in x]
words = list(set(all_text)) #array of all the words in the training data
word2ind = {word: index for index, word in enumerate(words)}
index_unk = len(word2ind)
unk = "$UNK$" #for the cases when the test data will have a new word
word2ind[unk] = index_unk
ind2word = {index: word for index, word in enumerate(words)}
ind2word[index_unk] = unk
labels = list(set([c for x in y for c in x])) #all the labels
label2ind = {label: (index + 1) for index, label in enumerate(labels)}
ind2label = {(index + 1): label for index, label in enumerate(labels)}
print 'Vocabulary size:', len(word2ind), len(label2ind)

maxlen = max([len(x) for x in X])
print 'Maximum sequence length:', maxlen


def encode(x, n): #maps the labels to one hot vector
    result = np.zeros(n)
    result[x] = 1
    return result

X_enc = [[word2ind[c] for c in x] for x in X]
max_label = max(label2ind.values()) + 1
y_enc = [[0] * (maxlen - len(ey)) + [label2ind[c] for c in ey] for ey in y]
y_enc = [[encode(c, max_label) for c in ey] for ey in y_enc]
X_train = pad_sequences(X_enc, maxlen=maxlen)
y_train = pad_sequences(y_enc, maxlen=maxlen)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=4*32, train_size=21*32, random_state=42)
# ------------------------------- reading test data -------------------------------------------------------------------
def checkForUnk(c): #if there is an unknown word in the test data
    if c not in word2ind:
        c  = "$UNK$"
    return word2ind[c]

all_x,X,y =createData(f_test)
X_test_enc = [[checkForUnk(c) for c in x] for x in X]
X_test = pad_sequences(X_test_enc, maxlen=maxlen)

# ------------------------------- defining the model -------------------------------------------------------------------
max_features = len(word2ind)
embedding_size = 500
hidden_size = 150
out_size = len(label2ind) + 1

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam')

# ------------------------------- training and saving the model -------------------------------------------------------------------
batch_size = 32
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15, validation_data=(X_val,y_val))
model.save("../model/bilstm",True,True)

# ------------------------------- evaluating the model -------------------------------------------------------------------
def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

model = load_model("../model/bilstm")
pr = model.predict_classes(X_train)
yh = y_train.argmax(2)
fyh, fpr = score(yh, pr)
print 'Training accuracy:', accuracy_score(fyh, fpr)
print 'Training confusion matrix:'
print confusion_matrix(fyh, fpr)
precision_recall_fscore_support(fyh, fpr) #evaluation on the training set


pr = model.predict_classes(X_test)

# ------------------------------- evaluating olia triples -------------------------------------------------------------------

predictions = []
pred = []
for i in range(0, len(X_test_enc)):
    x = X_test_enc[i]
    p = pr[i][-len(x):]
    for j in range(0, len(x)):
        word = ind2word[x[j]]
        label = "UNK"
        if (p[j] in ind2label):
            label = ind2label[p[j]]
        pred.append(label)
    predictions.append(pred)
    pred = []

def triple_score(predictions, y):
    truePositve = 0.0
    falsePositive = 0.0
    for k in range(0,len(predictions)):
        pred = predictions[k]
        y_gold = y[k]
        for i in range(0,len(pred)):
            pr = re.split("\\|",pred[i])[1:]
            y_tr = re.split("\\|",y_gold[i])[1:]
            for p in pr:
                if "MorphosyntacticCategory" not in p:
                    if p in y_tr:
                        truePositve=truePositve+1.0
                    else:
                        falsePositive=falsePositive+1.0
    precision = truePositve/(truePositve+falsePositive)
    print "precision",precision




triple_score(predictions,y)
print 'Testing accuracy:', triple_score(predictions,y)
