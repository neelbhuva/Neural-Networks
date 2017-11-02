import math
#from sklearn import svm
import numpy as np
import pandas as pd
#from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from svmutil import *

def readData(fd):
	d = []
	x = []
	for line in fd:
		t = {}
		index = []
		#split on spaces
		data = line.split()
		#first element in data is class label (desired output)
		d.append(int(data[0]))
		#loop through all inputs, data[0] is class label
		for i in data[1:len(data)]:
			temp = i.split(":")
			t[int(temp[0])] = float(temp[1]) # temp[0] is index and temp[1] is value as a string
			index.append(int(temp[0]))
		keys = t.keys()
		for i in range(1,9):
			if(i in keys):
				continue
			else:
				#index i is missing, set its value to zero
				t[i] = float(0)
		x.append(t)
	#x is a list of dictionaries that contain input vector
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

#train a linear svm on entire train set
def libsvm(train_data,d_train,test_data,d_test,C):
	print("------------------------------------------------")
	print("Training linear svm using entire training set ...")
	y_pred = []
	acc = []
	best_C = 0
	best_accuracy = 0
	for i in C:
		param = svm_parameter('-t 0 -c ' + str(i) + ' -q')
		# param.kerneltype = LINEAR
		# param.C = i
		prob = svm_problem(d_train,train_data)
		m = svm_train(prob, param,'-q')
		(p_lbl, p_acc, p_prob) = svm_predict(d_test,test_data,m)
		#y_pred = m.predict(test_data)
		#print(p_acc[0])
		#y_pred = m.predict_values(test_data)
		# classifier = svm.SVC(C=i,kernel='linear')
		# classifier.fit(train_data,d_train)
		# y_pred = classifier.predict(test_data)
		# accuracy = accuracy_score(d_test, y_pred)
		acc.append(p_acc[0])
		if(p_acc[0] > best_accuracy):
			best_accuracy = p_acc[0]
			best_C = i
	my_xticks = C
	x = [1,2,3,4,5,6,7,8,9,10,11,12,13]
	plt.xticks(x, my_xticks)
	plt.plot(x,acc,marker='o')
	plt.title('Training with linear svm')
	plt.ylabel('Accuracy')
	plt.xlabel('Parameter C')
	plt.savefig("Accuracy_vs_C.png")
	print("Best C : " + str(best_C))
	print("Best accuracy : " + str(best_accuracy))
	print("------------------------------------------------")
	return (best_C,best_accuracy)

def partitionTrainSet(train_data,d_train,n):
	train_data = pd.DataFrame(train_data)
	d_train = pd.DataFrame(d_train)
	d_train.columns = ['y']
	train_data = pd.concat([train_data,d_train],axis = 1)
	#shuffle all the instances
	train_data = train_data.sample(frac=1)
	#get half (n = 2) of the training instances
	train_set = train_data.iloc[:int(train_data.shape[0]/n)]
	#shuffle the instances again
	#train_set = train_set.sample(frac=1)
	return train_set

def crossValidateRBF(train_data,d_train,test_data,d_test,C,alpha):
	print("------------------------------------------------")
	print("Crossvalidating using 50 percent training set...")
	best_C = 0
	best_alpha = 0
	best_C_list = []
	best_alpha_list = []
	best_accuracy = 0
	train_set = partitionTrainSet(train_data,d_train,2)
	#print(train_set)
	#split the train set into 5 equal parts
	df_train = np.array_split(train_set,5)
	# print(df_train[0].head())
	# print(df_train[1].head())
	input_columns = [1,2,3,4,5,6,7,8]
	acc = []	
	for j in C:
		for k in alpha:
			t = []
			for i in range(0,len(df_train)):
				param = svm_parameter('-t 2 -c ' + str(j) + ' -g ' + str(k) + ' -q')
				# param.kerneltype = RBF
				# param.C = j
				# param.gamma = k				
				#classifier = svm.SVC(C=j,kernel='rbf',gamma=k)
				df_cross_train = pd.DataFrame(columns=[1,2,3,4,5,6,7,8,'y'])
				df_cross_test = pd.DataFrame(columns=[1,2,3,4,5,6,7,8,'y'])
				#print(df_cross_train)
				#select one of 5 sets as test set every time.
				for p in range(0,5):
					#print("i : " + str(i) + " p : " + str(p))
					if(not(p == i)):
						df_cross_train = pd.concat([df_cross_train,df_train[p]],axis=0)
					else:
						df_cross_test = pd.concat([df_cross_test,df_train[i]],axis=0)
				# print("-------")
				# print(df_cross_train.index)
				# print(df_cross_test.index)
				prob = svm_problem(df_cross_train['y'].values.tolist(),df_cross_train[input_columns].values.tolist())
				m = svm_train(prob, param,'-q')
				p_lbl, p_acc, p_prob = svm_predict(df_cross_test['y'].values.tolist(),df_cross_test[input_columns].values.tolist(),m,'-q')
				#y_pred = m.predict(test_data)
				#print(p_acc[0])
				# classifier.fit(df_cross_train[input_columns],df_cross_train['y'])
				# y_pred = classifier.predict(df_cross_test[input_columns])			
				#t.append(accuracy_score(df_cross_test['y'], y_pred))
				t.append(p_acc[0])
			accuracy = np.average(t)
			print(j,k,accuracy)
			if(accuracy > best_accuracy):
					best_accuracy = accuracy
					best_C = j
					best_alpha = k
					best_C_list.append(j)
					best_alpha_list.append(k)
			acc.append(accuracy)
	acc = np.reshape(acc,(len(C),len(alpha)))
	acc = pd.DataFrame(data=acc)
	acc.columns = alpha
	#acc.rows = C
	acc.to_csv("Accuracy_matrix.csv")
	print("Best C : " + str(best_C))
	print("Best alpha : " + str(best_alpha))
	print("Best Accuracy : " + str(best_accuracy))
	print("------------------------------------------------")
	return (best_C,best_alpha,best_C_list,best_alpha_list,best_accuracy)

def trainWithEntireSet(train_data,d_train,test_data,d_test,best_C,best_alpha):
	print("------------------------------------------------")
	print("Training with entire train set...")
	best_accuracy = 0
	# best_C = 0
	# best_alpha = 0
	for i in range(0,1):
		param = svm_parameter('-t 2 -c ' + str(best_C) + ' -g ' + str(best_alpha) + ' -q')
		# param.kerneltype = RBF
		# param.C = best_C
		# param.gamma = best_alpha
		prob = svm_problem(d_train,train_data,'-q')
		m = svm_train(prob, param)
		p_lbl, p_acc, p_prob = svm_predict(d_test,test_data,m)
		accuracy = p_acc[0]
		# classifier = svm.SVC(C=best_C,kernel='rbf',gamma=best_alpha)
		# classifier.fit(train_data,d_train)
		# y_pred = classifier.predict(test_data)
		# accuracy = accuracy_score(d_test, y_pred)
		if(accuracy > best_accuracy):
			best_accuracy = accuracy
			best_C = best_C
			best_alpha = best_alpha
	print("Best C : " + str(best_C))
	print("Best alpha : " + str(best_alpha))
	print("Best Accuracy : " + str(best_accuracy))
	print("------------------------------------------------")
	return best_accuracy

if __name__ == '__main__':
	#read train and test data
	f_train = open('train.txt','r')
	f_test = open('test.txt','r')
	(train_data,d_train) = readData(f_train)	
	(test_data,d_test) = readData(f_test)
	#print(train_data[0])
	C = [math.pow(2,-4)]
	for i in range(1,13):
		C.append(C[0]*math.pow(2,i))
	#print(C)
	#train_data = transformInput(train_data)
	#test_data = transformInput(test_data)
	#libsvm(train_data,d_train,test_data,d_test,C)
	(best_C,best_accuracy) = libsvm(train_data,d_train,test_data,d_test,C)
	alpha = C
	(best_C,best_alpha,best_C_list,best_alpha_list,best_accuracy) = crossValidateRBF(train_data,d_train,test_data,d_test,C,alpha)
	#print(best_C_list,best_alpha_list)
	accuracy = trainWithEntireSet(train_data,d_train,test_data,d_test,best_C,best_alpha)
	#print("Accuracy : " + str(accuracy))