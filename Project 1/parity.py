import itertools
import numpy as np
import math
import random as rd

def initialize_weights():
	weights = []
	for i in range(0,num_units):
		weights.append([])
		for j in range(0,4):
			rand = rd.random()
			flag = rd.choice([0,1])
			if(flag == 0):
				rand = rand * (-1)
			weights[i].append(rand)
	return weights

def initialize_bias():
	weights = []
	for i in range(0,num_units):
		rand = rd.random()
		flag = rd.choice([0,1])
		if(flag == 0):
			rand = rand * (-1)
		weights.append(rand)
	return weights

def get_sigmoid(x,a):
	return 1 / (1 + math.exp((-1) * a * x))

def get_sigmoid_prime(x,a):
	sig = get_sigmoid(x,a)
	return (a * sig * (1 - sig))

def get_v_of_n(x,w,b):
	v_n = np.dot(x,w)
	v_n = b + v_n
	return v_n

def desiredOutputs(lst):
	des_output = []
	for i in lst:
		s = 0
		s = sum(i)
		if(s % 2 == 0):
			des_output.append(0)
		else:
			des_output.append(1)
	return des_output

def activate(v_n):
	#print("In activate")
	y_n = []
	for i in v_n:
		#print(i)
		y_n.append(get_sigmoid(i,1))
	return y_n

def get_error(d_n,y_n):
	return (d_n - y_n)

def get_delta_k(error_out,v_n_out):
	return (error_out * get_sigmoid_prime(v_n_out,1))

def out_weight_update(w_n,b_n,delta_w_n,delta_b_n,learn_rate,error_out,v_n_out,y_n,alpha):
	w_n_plus_one = []
	b_n_plus_one = 0
	delta_w_n_new = []
	delta_b_n_new = 0
	for i in range(0,len(w_n)):
		w_n_plus_one.append(w_n[i] + learn_rate * get_delta_k(error_out,v_n_out) * y_n[i] + delta_w_n[i])
		delta_w_n_new.append(alpha * (w_n_plus_one[i] - w_n[i]))
	b_n_plus_one = b_n + learn_rate * get_delta_k(error_out,v_n_out) * 1 + delta_b_n
	delta_b_n_new = alpha * (b_n_plus_one - b_n)
	return (w_n_plus_one,b_n_plus_one,delta_w_n_new,delta_b_n_new)

def initializeWithZero(m,n):
	temp = []
	for i in range(0,m):
		temp.append([])
		for j in range(0,n):
			temp[i].append(0)
	return temp

def hidden_weight_update(w_n,b_n,delta_w_n,delta_b_n,learn_rate,v_n,x,delta_k,alpha):
	w_n_plus_one = []
	b_n_plus_one = []
	delta_w_n_new = initializeWithZero(num_hidden_units,len(w_n[0]))
	delta_b_n_new = []
	for i in range(0,num_hidden_units):
		w_n_plus_one.append([])
		for j in range(0,len(w_n[i])):
			w_n_plus_one[i].append(w_n[i][j] + learn_rate * get_sigmoid_prime(v_n[i],1) * delta_k * w_n[4][i] * x[j] + delta_w_n[i][j])
			delta_w_n_new[i][j] = alpha * (w_n_plus_one[i][j] - w_n[i][j])
		b_n_plus_one.append(b_n[i] + learn_rate * get_sigmoid_prime(v_n[i],1) * delta_k * w_n[4][i] * 1 + delta_b_n[i])
		delta_b_n_new.append(alpha * (b_n_plus_one[i] - b_n[i]))
	return(w_n_plus_one,b_n_plus_one,delta_w_n_new,delta_b_n_new)

def isDesiredError(error,stop_error):
	for i in error:
		if(i > stop_error):
			return False
	return True

def initializeWithOne(size):
	temp = []
	for i in range(0,size):
		temp.append(1)
	return temp

def update_weights(weights,b,w_n_plus_one,b_n_plus_one,w,b1):
	for i in range(0,len(w_n_plus_one)):
		weights[4][i] = w_n_plus_one[i]
	b[4] = b_n_plus_one
		
	for i in range(0,num_hidden_units):
		for j in range(0,4):
			weights[i][j] = w[i][j]
		b[i] = b1[i]
	return (weights,b)

def write_to_file(fd,learn_rate,epochs,error):
	fd.write("Learning Rate : " + str(learn_rate) + "\n")
	fd.write("Number of epochs : " + str(epochs) + "\n")
	fd.write("Error : " + str(error) + "\n")

def learn(train_data,des_output,learn_rate,weights,b,fd,alpha):
	stop_error = 0.05
	size = 16
	error = initializeWithOne(size)
	epochs = 0	
	print("Learning Rate : " + str(learn_rate))
	print("alpha : " + str(alpha))
	#print(weights)
	while(not(isDesiredError(error,stop_error))):
		delta_b_n_out = 0
		delta_w_n_out = [0,0,0,0]
		delta_w_n = initializeWithZero(num_hidden_units,4)
		delta_b_n = [0,0,0,0]
		#train_data = rd.sample(train_data,len(train_data))
		#print(train_data)
		for p in range(0,len(train_data)):
			#print("---------------------------Training data : " + str(p))
			v_n = []
			v_n_out = []
			for i in range(0,num_hidden_units):
				v_n.append(get_v_of_n(train_data[p],weights[i],b[i]))
			y_n = activate(v_n)			
			#print(y_n)
			for i in range(num_hidden_units,num_hidden_units + num_out_units):
				v_n_out.append(get_v_of_n(y_n,weights[4],b[4]))
			y_n_out = activate(v_n_out) 
			#print(y_n_out)
			error_out = get_error(des_output[p],y_n_out[0])
			error[p] = abs(error_out)
			w_n_plus_one,b_n_plus_one,delta_w_n_out,delta_b_n_out = out_weight_update(weights[4],b[4],delta_w_n_out,delta_b_n_out,learn_rate,error_out,v_n_out[0],y_n,alpha)
			
			delta_k = get_delta_k(error_out,v_n_out[0])
			w,b1,delta_w_n,delta_b_n = hidden_weight_update(weights,b,delta_w_n,delta_b_n,learn_rate,v_n,train_data[p],delta_k,alpha)

			weights,b = update_weights(weights,b,w_n_plus_one,b_n_plus_one,w,b1)
			
		epochs = epochs + 1
		if(epochs % 10000 == 0):
			print(epochs)
		if(epochs % 15000 == 0):
			print("Error : " + str(error))
	
	write_to_file(fd,learn_rate,epochs,error)	
	# print(weights)
	# print(b)

def initializeWith(x,m,n):
	temp = []
	if(n == 0):
		for i in range(0,m):
			temp.append(x[i])
	else:
		for i in range(0,m):
			temp.append([])
			for j in range(0,n):
				temp[i].append(x[i][j])
	return temp

if __name__ == '__main__':
	global num_hidden_units
	num_hidden_units = 4
	global num_out_units
	num_out_units = 1
	global num_units
	num_units = num_out_units + num_hidden_units
	n = 4
	lst = list(itertools.product([0, 1], repeat=n))
	des_output = desiredOutputs(lst)
	#print(lst)
	#print(des_output)
	global initial_weights
	#initial_weights = initialize_weights()
	#initial_weights = [[-0.6360791436676354,0.6004796194650841,-0.956914874387614,0.12116993046736135][0.5022367278828641,0.5646154529044839,0.5051043640099573,-0.5965247238641179],[0.25779655410004676,0.8457361770967498,0.5231879825256082, -0.8408321706937136],[-0.28164521687268285, -0.24731780476998655, 0.24581986297639502, -0.9253982569254791],[0.45015554119658063,-0.2811335040976013,-0.1821990011957696,0.8583247394747876]]
	initial_weights = np.random.rand(5,4)
	#print(init_weights)
	global initial_b
	#initial_b = initialize_bias()
	#initial_b = [-0.16357184125119684,0.36218742680041827,0.9068696623198483,0.983684804487359,-0.888105722149733]
	initial_b = np.random.rand(5)
	#print(init_b)
	fd = open("Results.txt",'w')
	learn_rate = np.arange(0.05,0.51,0.05)
	alpha = 0
	print(learn_rate)
	fd.write("Initial Weights : " + str(initial_weights))
	fd.write("Initial bias : " + str(initial_b))
	for i in learn_rate:
		init_weights = initializeWith(initial_weights,num_units,4)
		init_b = initializeWith(initial_b,num_units,0)
		print("Initial Weights : ")
		print(init_weights)
		print("Initial bias : ")
		print(init_b)
		#learn(lst,des_output,i,init_weights,init_b,fd)
		learn(lst,des_output,i,init_weights,init_b,fd,alpha)
	fd.close()

	fd = open("Results_momentum.txt",'w')
	fd.write("Initial Weights : " + str(initial_weights))
	fd.write("Initial bias : " + str(initial_b))
	alpha = 0.9
	for i in learn_rate:
		init_weights = initializeWith(initial_weights,num_units,4)
		init_b = initializeWith(initial_b,num_units,0)
		print("Initial Weights : ")
		print(init_weights)
		print("Initial bias : ")
		print(init_b)
		#learn(lst,des_output,i,init_weights,init_b,fd)
		learn(lst,des_output,i,init_weights,init_b,fd,alpha)
	#learn(lst,des_output,learn_rate,init_weights,init_b,fd)
	fd.close()