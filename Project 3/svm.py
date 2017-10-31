import math
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def readData(fd):
	d = []
	x = []
	for line in fd:
		t = {}
		index = []
		data = line.split()
		d.append(data[0])
		for i in data[1:len(data)]:
			temp = i.split(":")
			t[int(temp[0])] = float(temp[1])
			index.append(int(temp[0]))
		keys = t.keys()
		for i in range(1,9):
			if(i in keys):
				continue
			else:
				t[i] = float(0)
		x.append(t)
	#print(d)
	#print(x[0])
	return (x,d)

def transformInput(data):
	t = []
	for i in range(0,len(data)):
		t.append([])
		for key,value in data[i].items():
			t[i].append(value)
	t = np.reshape(t,(len(data),8))
	#print(t)
	return t

def libsvm(train_data,d_train,test_data,d_test,C):
	y_pred = []
	acc = []
	for i in C:
		classifier = svm.SVC(C=i,kernel='linear')
		classifier.fit(train_data,d_train)
		y_pred = classifier.predict(test_data)
		acc.append(accuracy_score(d_test, y_pred))
	#print(y_pred)
	print(acc)
	plt.plot(acc)
	plt.show()

if __name__ == '__main__':
	#read train and test data
	f_train = open('train.txt','r')
	f_test = open('test.txt','r')
	(train_data,d_train) = readData(f_train)
	
	(test_data,d_test) = readData(f_test)
	C = [math.pow(2,-4)]
	for i in range(1,13):
		C.append(C[0]*math.pow(2,i))
	#print(C)
	train_data = transformInput(train_data)
	test_data = transformInput(test_data)
	libsvm(train_data,d_train,test_data,d_test,C)