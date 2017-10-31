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
	print("Training linear svm using entire training set ...")
	y_pred = []
	acc = []
	best_C = 0
	best_accuracy = 0
	for i in C:
		classifier = svm.SVC(C=i,kernel='linear')
		classifier.fit(train_data,d_train)
		y_pred = classifier.predict(test_data)
		accuracy = accuracy_score(d_test, y_pred)
		acc.append(accuracy)
		if(accuracy > best_accuracy):
			best_accuracy = accuracy
			best_C = i
	plt.plot(acc)
	plt.show()
	print("Best C : " + str(best_C))
	print("Best accuracy : " + str(best_accuracy))
	return (best_C,best_accuracy)

def partitionTrainSet(train_data,d_train,n):
	train_data = pd.DataFrame(train_data)
	d_train = pd.DataFrame(d_train)
	d_train.columns = ['y']
	train_data = pd.concat([train_data,d_train],axis = 1)
	train_data = train_data.sample(frac=1)
	train_set = train_data.iloc[:int(train_data.shape[0]/n)]
	train_set = train_set.sample(frac=1)
	return train_set

def crossValidateRBF(train_data,d_train,test_data,d_test,C,alpha):
	print("Crossvalidating using 50 percent training set...")
	best_C = 0
	best_alpha = 0
	best_accuracy = 0
	train_set = partitionTrainSet(train_data,d_train,2)
	df_train = np.array_split(train_set,5)
	input_columns = [0,1,2,3,4,5,6,7]
	acc = []
	t = []
	for j in C:
		for k in alpha:
			for i in range(0,len(df_train)):
				classifier = svm.SVC(C=j,kernel='rbf',gamma=k)
				df_cross_train = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,'y'])
				df_cross_test = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,'y'])
				for p in range(0,5):
					if(not(p == i)):
						df_cross_train = pd.concat([df_cross_train,df_train[i]],axis=0)
					else:
						df_cross_test = pd.concat([df_cross_test,df_train[i]],axis=0)
				#print(df_cross_test)
				classifier.fit(df_cross_train[input_columns],df_cross_train['y'])
				y_pred = classifier.predict(df_cross_test[input_columns])			
				t.append(accuracy_score(df_cross_test['y'], y_pred))
			accuracy = np.average(t)
			if(accuracy > best_accuracy):
					best_accuracy = accuracy
					best_C = j
					best_alpha = k
			acc.append(accuracy)
	acc = np.reshape(acc,(len(C),len(alpha)))
	acc = pd.DataFrame(data=acc)
	acc.columns = alpha
	acc.rows = C
	print(acc)
	print("Best C : " + str(best_C))
	print("Best alpha : " + str(best_alpha))
	print("Best Accuracy : " + str(best_accuracy*100))
	return (best_C,best_alpha,best_accuracy)

def trainWithEntireSet(train_data,d_train,test_data,d_test,best_C,best_alpha):
	print("Training with entire train set...")
	classifier = svm.SVC(C=best_C,kernel='rbf',gamma=best_alpha)
	classifier.fit(train_data,d_train)
	y_pred = classifier.predict(test_data)
	accuracy = accuracy_score(d_test, y_pred)
	return accuracy

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
	(best_C,best_accuracy) = libsvm(train_data,d_train,test_data,d_test,C)
	alpha = C
	(best_C,best_alpha,best_accuracy) = crossValidateRBF(train_data,d_train,test_data,d_test,C,alpha)
	accuracy = trainWithEntireSet(train_data,d_train,test_data,d_test,best_C,best_alpha)
	print("Accuracy : " + str(accuracy*100))