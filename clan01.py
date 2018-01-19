# -*- coding: utf-8 -*-
import requests
import numpy as np
import json

train_data = json.load(open('/home/baskar/Documents/coc/clan_train_dataset.json')) # train dataset 
test_data = json.load(open('/home/baskar/Documents/coc/clan_test_dataset.json')) # test dataset

log = []
log_t = []

for i in train_data['items']:
	#print i['result'],' ',i['teamSize'],' ',i['clan']['clanLevel'],' ',i['clan']['stars'],' ',i['clan']['destructionPercentage'],' ',i['opponent']['tag'],' ',i['opponent']['clanLevel'],' ',i['opponent']['stars'],' ',i['opponent']['destructionPercentage']
	if i['result'] == 'win':
		result = 1.0
	else:
		result = 0.0
	log.append([result,i['clan']['destructionPercentage'],i['clan']['clanLevel'],i['clan']['stars'],i['clan']['attacks'],i['clan']['expEarned'],i['teamSize'],i['opponent']['clanLevel'],i['opponent']['stars'],i['opponent']['destructionPercentage']])

for i in test_data['items']:
	#print i['result'],' ',i['teamSize'],' ',i['clan']['clanLevel'],' ',i['clan']['stars'],' ',i['clan']['destructionPercentage'],' ',i['opponent']['tag'],' ',i['opponent']['clanLevel'],' ',i['opponent']['stars'],' ',i['opponent']['destructionPercentage']
	if i['result'] == 'win':
		result = 1.0
	else:
		result = 0.0
	log_t.append([result,i['clan']['destructionPercentage'],i['clan']['clanLevel'],i['clan']['stars'],i['clan']['attacks'],i['clan']['expEarned'],i['teamSize'],i['opponent']['clanLevel'],i['opponent']['stars'],i['opponent']['destructionPercentage']])

lst = [item[0] for item in log_t]
#print "test_data: ",lst
a = np.array(log)
a_t = np.array(log_t)
	
print "shape:",a_t.shape,"len:",len(a_t)
#print 'result: ',a[29:58:,0:2]
#print 'input_data:',a[0:15:,1:7]
#print '\n' 
#print 'input_data:',a[5:10:,1:4]
#print '\n'
#print 'input_data:',a[10:15:,1:4]  
#print a[:]
###################################################################
y = len(a_t)
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

trainX = a[0:58:,1:10]
trainY = a[0:58:,0:1]
testX = a_t[0:y,1:10]
testY = a_t[0:y,0:1]
log_e = []
# create model
model = Sequential()
model.add(Dense(6, input_dim=9, kernel_initializer='uniform', activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(3, kernel_initializer='uniform', activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
print "PROCESSING..."
model.fit(trainX, trainY, epochs=2000, batch_size=10,  verbose=1)
# calculate predictions
predictions = model.predict(testX)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
log_e.append(rounded)
accuracy = model.evaluate(testX,testY,verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], accuracy[1]*100))

log_s = []
log_d = []
log_f = []
for i in range(y):
	log_d.append(a_t[0:i,0:1]);

for i in predictions:
	log_s.append(round(i))
	log_f.append(i)

print "\nOUTPUT"
for i in range(y):
	print "actual output:",lst[i],"......predicted output:",log_s[i]
print("%s: %.2f%%" % (model.metrics_names[1], accuracy[1]*100))
