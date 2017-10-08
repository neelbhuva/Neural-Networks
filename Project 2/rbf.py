import numpy as np
import math
import random as rd
import matplotlib.pyplot as plt
import pandas as pd
import copy

def initialize_weights(k):
	weights = []
	for i in range(0,k):
		rand = rd.random()
		flag = rd.choice([0,1])
		if(flag == 0):
			rand = rand * (-1)
		weights.append(rand)
	return weights

def sampleDataPoints(size):
	input_x = []
	output_d = []
	noise_interval = ((-1)*0.1,0.1000000000000000001) #0.11 because number less than 0.11 are generated 
	input_interval = (0,1.00000000001)
	for i in range(0,size):
		noise = np.random.uniform(noise_interval[0],noise_interval[1])
		input_data = np.random.uniform(input_interval[0],input_interval[1])
		#print(noise,input_data)
		input_x.append(input_data)
		output_d.append(0.5 + 0.4 * math.sin(2 * math.pi * input_data))
	return (input_x,output_d)

def getMeanUsingIndex(x,indices):
	total = 0
	for i in indices:
		total = total + x[i]
	return (total/len(indices))

def anyCenterChanged(k_centers,new_centers):
	for i in range(0,len(k_centers)):
		if(k_centers[i] != new_centers[i]):
			return True
	return False

def assignInputToClusters(x,d,k_centers):
	#keep track of clusters
	k_clusters = {}
	# print("Centers : ")
	# print(k_centers)
	dissimilarity = []
	#assign all input patterns to clusters
	for key,i in k_centers.items():
		for j in x:
			dissimilarity.append((i-j)*(i-j)) #eucledian similarity

	dissimilarity = np.reshape(dissimilarity,(len(k_centers),len(x)))
	df_dissim = pd.DataFrame(data=dissimilarity)
	#print(df_dissim)
	for i in range(0,df_dissim.shape[0]):
		k_clusters[i] = [];
	for i in df_dissim.columns:
		#print(df_dissim.nsmallest(1,i).index[0],end = " ")
		shortest_dist_index = df_dissim.nsmallest(1,i).index[0]
		k_clusters[shortest_dist_index].append(i)
	#print(k_clusters)
	new_centers = {}
	for key,value in k_clusters.items():
 		#print(key,value)
		new_centers[key]  = getMeanUsingIndex(x,value)
	# print("New Centers : ")
	# print(new_centers)
	if(anyCenterChanged(k_centers,new_centers)):
		assignInputToClusters(x,d,new_centers)
	return (new_centers,k_clusters)

def printDataUsingIndex(x,indices):
	data = []
	for i in indices:
		data.append(x[i])
	print(data)

def varianceUsingIndex(x,indices):
	variance = []
	data = []
	for i in indices:
		data.append(x[i])
	return (np.var(data))

def kMeans(x,d,K):
	#select K number of input patterns as cluster centers randomly
	centers = rd.sample(x,K)
	k_centers = {}
	for i in range(0,len(centers)):
		k_centers[i] = centers[i]

	(k_centers,k_clusters) = assignInputToClusters(x,d,k_centers)
	variance = {}
	temp = []
	one_point_clusters = []
	for key,value in k_clusters.items():
		if(len(value) == 1):
			one_point_clusters.append(key)
		else:
			variance[key] = varianceUsingIndex(x,value)
			temp.append(variance[key])
		#printDataUsingIndex(x,value)
	for i in one_point_clusters:
		variance[i] = np.mean(temp)
	# print("Variance : ")
	# print(variance)
	return (k_clusters,k_centers,variance)

def computeGuassian(input_vec,var,center):
	eucledian = (input_vec-center) * (input_vec-center)
	a = (-1)/(2 * var)
	return np.exp(a * eucledian)

def getFx(weights,guassian):
	total = 0
	#print(weights,guassian)
	for key,value in guassian.items():
		#print("\nWeights[key] : " + str(weights[key]))
		total = total + (weights[key] * value)
	return total 

def weightUpdate(weights,d,y,input_signal,learn_rate):
	w_n_plus_one = []
	#print("\nWeight length : " + str(len(weights)))
	for i in range(0,len(weights)):
		#print(weights[i],learn_rate,(d-y),input_signal[i])
		w_n_plus_one.append(weights[i] + (learn_rate * (d-y) * input_signal[i]))
	return w_n_plus_one

def plotData(input_patterns,d,weights,k_clusters,k_centers,k_variance,k,learn_rate):
	y = []
	for i in range(0,len(input_patterns)):
		guassian = {}
		for key,value in k_clusters.items():
			guassian[key] = computeGuassian(input_patterns[i],k_variance[key],k_centers[key])
		y.append(getFx(weights,guassian))
	plt.scatter(input_patterns, y, marker='.',label="actual output")
	plt.scatter(input_patterns, d, marker='*',label="sampled output")
	plt.title("K = " + str(k) + " and Learning Rate = " + str(learn_rate))
	plt.legend()
	#plt.show()
	plt.savefig("K_" + str(k) + "_learn_rate_" + str(learn_rate) + ".png")
	plt.clf()

def computeFixedGuassianWidth(k_centers):
	temp_centers = []
	for key,value in k_centers.items():
		temp_centers.append(value)
	temp = sorted(temp_centers)
	return math.pow((temp[len(temp)-1]-temp[0]),2)

def RBF(input_patterns,d,bases,learn_rate,max_epochs,fixedGuassianWidth):
	(k_clusters,k_centers,k_variance) = kMeans(input_patterns,d,bases)
	weights = initialize_weights(bases)
	fixed_var = 0
	if(fixedGuassianWidth == True):
		fixed_var = computeFixedGuassianWidth(k_centers) / math.sqrt(2 * bases)
		fixed_var = math.pow(fixed_var,2)
		print("Fixed width : " + str(fixed_var))
	# print("\n\n\nInitial Weights : ")
	# print(weights)
	epochs = 0
	while(epochs <= max_epochs):
		for i in range(0,len(input_patterns)):
			guassian = {}
			for key,value in k_clusters.items():
				if(fixedGuassianWidth == True):
					guassian[key] = computeGuassian(input_patterns[i],fixed_var,k_centers[key])
				else:
					guassian[key] = computeGuassian(input_patterns[i],k_variance[key],k_centers[key])
			# print("\nGuassian : ")
			# print(guassian)
			Fx = getFx(weights,guassian)
			#print("\nFx : " + str(Fx) + "\n")
			weights = copy.copy(weightUpdate(weights,d[i],Fx,guassian,learn_rate))
			# print("\nIntermediate Weights : ")
			# print(weights)
		epochs = epochs + 1
		if(epochs % 50 == 0):
			print("\nEpochs : " + str(epochs) + "\n")
	# print("\nFinal Weights : ")
	# print(weights)
	plotData(input_patterns,d,weights,k_clusters,k_centers,k_variance,bases,learn_rate)

if __name__ == '__main__':
	numDataPoints = 75
	fixedGuassianWidth = True
	#input vector x, output scalar d
	(x,d) = sampleDataPoints(numDataPoints)
	# #plot the data points to verify visually
	# plt.scatter(x,d)
	# plt.show()
	#print(x,d)
	bases_series = [2,4,7,11,16]
	learn_rate_series = [0.01,0.02]
	max_epochs = 100
	for learn_rate in learn_rate_series:
		for bases in bases_series:
			print("---------------------------------------------------------------")
			print("Learn rate : " + str(learn_rate) + " # of bases : " + str(bases))
			RBF(x,d,bases,learn_rate,max_epochs,fixedGuassianWidth)
			print("---------------------------------------------------------------\n\n\n")