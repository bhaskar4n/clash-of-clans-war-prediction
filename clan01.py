# -*- coding: utf-8 -*-
import requests
import numpy as np
import json
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
train_data = json.load(open('/home/baskar/Documents/coc/train_dataset_warlog.json')) # train dataset 
log = []
log_t = []
for i in train_data['items']:
	if i['result'] == 'win':
		result = 1.0
	else:
		result = 0.0
	log.append([result,i['clan']['clanLevel'],i['clan']['attacks'],i['clan']['expEarned'],i['teamSize'],i['opponent']['clanLevel']])

lst = [item[0] for item in log]
a = np.array(log)
print "shape:",a.shape,"len:",len(a)
y = len(a)
xi = int(y/2)
seed = 7
np.random.seed(seed)
trainX = a[xi:y:,1:6]
trainY = a[xi:y:,0:1]
testX = a[0:xi,1:6]
testY = a[0:xi,0:1]
log_e = []
# create model
model = Sequential()
model.add(Dense(16, input_dim=5, kernel_initializer='uniform', activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
print "PROCESSING..."
model.fit(trainX, trainY, epochs=1000, batch_size=10,  verbose=1)
# calculate predictions
predictions = model.predict(testX)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
log_e.append(rounded)
accuracy = model.evaluate(testX,testY,verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], accuracy[1]*100))
log_s = []
log_d = []
log_f = []
for i in range(xi):
	log_d.append(a[0:i,0:1]);
for i in predictions:
	log_s.append(round(i))
	log_f.append(i)
print "train dataset"
for i in range(xi,y):
	print a[i:y:,1:6]
print "test dataset"
for i in range(0,xi):
	print a[0:i:,1:6]
print "\nOUTPUT"
for i in range(xi):
	print "data:",i,"...actual result",lst[i],"......predicted result:",log_s[i]
print("%s: %.2f%%" % (model.metrics_names[1], accuracy[1]*100))
