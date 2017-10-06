import numpy as np
import math
import random as rd
import matplotlib.pyplot as plt
import pandas as pd

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

def kMeans(x,d,K):
	#select K number of input patterns as cluster centers randomly
	k_centers = rd.sample(x,K)
	print(k_centers)
	similarity = []
	#assign all input patterns to clusters
	for i in k_centers:
		for j in x:
			similarity.append((i-j)*(i-j)) #eucledian similarity
	similarity = np.reshape(similarity,(len(k_centers),len(x)))
	df_sim = pd.DataFrame(data=similarity)
	print(similarity)

def RBF(x,d,bases,learn_rate,max_epochs):
	kMeans(x,d,bases)

if __name__ == '__main__':
	numDataPoints = 75
	#input vector x, output scalar d
	(x,d) = sampleDataPoints(numDataPoints)
	#plot the data points to verify visually
	# plt.scatter(x,d)
	# plt.show()
	bases_series = [2,4,7,11,16]
	learn_rate_series = [0.01,0.02]
	bases_series = [2]
	learn_rate_series = [0.01]
	max_epochs = 100
	for learn_rate in learn_rate_series:
		for bases in bases_series:
			RBF(x,d,bases,learn_rate,max_epochs)