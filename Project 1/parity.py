import itertools
import numpy as np
import math

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
	y_n = []
	for i in v_n:
		y_n.append(get_sigmoid(i,1))
	return y_n

def get_error(d_n,y_n):
	return d_n - y_n

def get_delta_k(error_out,v_n_out):
	return (error_out * get_sigmoid_prime(v_n_out,1))

def out_weight_update(w_n,b_n,learn_rate,error_out,v_n_out,y_n):
	w_n_plus_one = []
	b_n_plus_one = 0
	for i in range(0,len(w_n)):
		w_n_plus_one.append(w_n[i] + learn_rate * get_delta_k(error_out,v_n_out) * y_n[i])
	b_n_plus_one = b_n + learn_rate * get_delta_k(error_out,v_n_out) * get_sigmoid(1,1)
	return (w_n_plus_one,b_n_plus_one)

def hidden_weight_update(w_n,b_n,learn_rate,v_n,x,delta_k):
	w_n_plus_one = []
	b_n_plus_one = []
	for i in range(0,num_hidden_units):
		for j in range(0,len(w_n[i])):
			w_n_plus_one.append(w_n[i][j] + learn_rate * get_sigmoid_prime(v_n[i],1) * delta_k * w_n[4][i] * x[i])
		b_n_plus_one.append(b_n[i] + learn_rate * get_sigmoid_prime(v_n[i],1) * delta_k * b_n[4] * 1)
	return(w_n_plus_one,b_n_plus_one)

def learn(train_data,des_output,learn_rate,weights,b):
	stop_error = 0.05
	error = 1
	while(error > stop_error):
		for p in range(0,len(train_data)):
			v_n = []
			v_n_out = []
			for i in range(0,num_hidden_units):
				v_n.append(get_v_of_n(train_data[p],weights[i],b[i]))
			y_n = activate(v_n)
			#print(y_n)
			for i in range(num_hidden_units,num_hidden_units + num_out_units):
				v_n_out.append(get_v_of_n(y_n,weights[i],b[i]))
			y_n_out = activate(v_n_out) 
			#print(y_n_out)
			error_out = get_error(des_output[p],y_n_out[0])
			w_n_plus_one,b_n_plus_one = out_weight_update(weights[4],b[4],learn_rate,error_out,v_n_out[0],y_n)
			delta_k = get_delta_k(error_out,v_n_out[0])
			w,b1 = hidden_weight_update(weights,b,learn_rate,v_n,train_data[p],delta_k)
			weights[4] = w_n_plus_one
			b[4] = b_n_plus_one
			weights[1:3] = w
			b[1:3] = b1
		print(error)
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
	print(lst)
	print(des_output)
	v_n = get_v_of_n(lst[2],lst[3],1)
	print(v_n)
	learn_rate = 0.5
	init_weights = [(-1,-1,-1,0),(0,-1,-1,1),(1,1,0,-1),(0,0,1,1),(-1,0,-1,-1)]
	init_b = (0,1,1,-1,0)
	learn(lst,des_output,learn_rate,init_weights,init_b)