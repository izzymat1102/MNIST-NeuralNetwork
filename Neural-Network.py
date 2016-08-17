# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 02:15:09 2016

@author: izuki
"""
# ­*­ coding: utf­8 ­*­ """ Spyder Editor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def plot(iteration):
    iteration = np.arange(iteration)
    plt.plot(iteration,loss_record, label = "PF", color = "cyan")
    
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("loss")
    plt.show()

def CostFunction():
    #Cross-Entropy Cost
    #return np.sum(-(teacher*np.log(a4) + (1-teacher)*np.log(1-a4)))	
    #Softmax CostFunction
    return 0
def Prediction (feed):

    #Layer1
    p_a1 = np.array([feed]).T
    p_z2 = np.dot(W1,p_a1)
    #Layer2
    p_a2 = ReLU(p_z2)
    p_a2 = np.vstack([p_a2,1])
    p_z3 = np.dot(W2,p_a2)
    #Layer3    
    p_a3 = ReLU(p_z3)
    p_a3 = np.vstack([p_a3,1])
    p_z4 = np.dot(W3,p_a3)
    #Output
    p_a4 = softmax(p_z4)

    return np.argmax(p_a4)
    
def Test(testnum):
    count = 0
    nums = [0] * 10
    correct_prediction = [0] * 10
    pre_nums = [0] * 10
	
    for i in range(40000,40000+testnum):
        test_pixels = data.iloc[i,1:]
        test_label = data.iloc[i,0]

        inputto = np.array(test_pixels,dtype = float)/255
        if Prediction(inputto) == test_label: #Correct Prediction
            count += 1
            correct_prediction[test_label] += 1
        
        nums[test_label] += 1
        pre_nums[Prediction(inputto)] += 1
        
    accuracy = count/float(testnum)
    print 'Accuracy in percentage is {0}%'.format(accuracy*100) 
    print 'sample distribution is ', nums
    print 'prediction distribution is ', pre_nums
    print 'correct prediction distribution is ', correct_prediction
        	
	
def Load_Picture(row):
    #ith row
    
    global label, teacher, feeds
    
    label = data.iloc[row,0]
    teacher = np.zeros((10,1))
    teacher[label] = 1
    feeds = np.array(data.iloc[row,1:],dtype = float)/255
    
def Theta(in_len, out_len):
    l = float(in_len + out_len + 1)
    alpha = np.sqrt((6-(1-1/(1+np.exp(-l)))*6)/l)
    matrix = (np.random.rand(in_len, out_len) - 0.5) * alpha
    return matrix
    
def Weights_Initialization():
    
    global W1, W2, W3
    global dW1, dW2, dW3

    outputnum = 10
	
    #Weights Initialization in range 0~1
    W1 = Theta(h1nodes,784) 
    W2 = Theta(h2nodes,h1nodes+1)
    W3 = Theta(outputnum,h2nodes+1)

    #Delta Weights Initialization with zeros
    dW1 = np.zeros(W1.shape)
    dW2 = np.zeros(W2.shape)
    dW3 = np.zeros(W3.shape)
    
def sigmoid(vector):
    return 1 / (1 + np.exp(vector))
    
def ReLU(vector):
    return np.maximum(vector, 0, vector)

def softmaxx(vector):
    ex = np.exp(vector)
    ex /= float(sum(ex))
    return ex
def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T # ndim = 2 
def Forward_Propagation():
    
    global a1, a2, a3, a4
    global z2, z3, z4

    #Layer1
    a1 = np.array([feeds]).T
    z2 = np.dot(W1,a1)

    #Layer2
    a2 = ReLU(z2)
    a2 = np.vstack([a2,1])  #Add a bias term
    z3 = np.dot(W2,a2)
    #Layer3    
    a3 = ReLU(z3)
    a3 = np.vstack([a3,1])  #Add a bias term
    z4 = np.dot(W3,a3)
    #Output
    a4 = softmax(z4)

def dReLU(vector):
    dvec = np.maximum(vector,0)/vector
    dvec[np.isnan(dvec)] = 0 
    return dvec
    
def Back_Propagation():
     
    global dW1, dW2, dW3
    
    #error(delta) calculation    
    del4 = teacher - a4
    
    del3 = np.dot(W3.T, del4) * np.vstack([dReLU(z3),1])
    del3 = np.delete(del3, -1, 0) #Delete bias term(?), reduce the dimension
    
    del2 = np.dot(W2.T, del3) * np.vstack([dReLU(z2),1])
    del2 = np.delete(del2, -1, 0)
    
    #Obtain the partial derivative of the weights
    dW3 += np.dot(del4, a3.T)
    dW2 += np.dot(del3, a2.T)
    dW1 += np.dot(del2, a1.T)

def Weights_Update(mu):

    global W3, W2, W1
    W3 -= mu * dW3/batch_size     # mu = learning_rate
    W2 -= mu * dW2/batch_size
    W1 -= mu * dW1/batch_size
    
def Learning(iteration,learning_rate,iter_count = 0):

    global loss_record
    loss_record = np.array([])

    Weights_Initialization()            #Set weights
    for j in range(iteration):		#Iteration
        loss = 0
		
        for i in range(batch_size):	   #loop through a batch          
            Load_Picture(i)
            Forward_Propagation()
            Back_Propagation()
            loss += CostFunction()
        Weights_Update(learning_rate) 
           
        loss /= batch_size
        iter_count += 1
        print 'iteration {0} loss: {1}'.format(iter_count, loss)
        if loss != loss:
            sys.exit("Softmax Overflow")
            
        loss_record = np.append(loss_record,loss)
  
    print 'Learning Completed'


### Main ###

#Load data
#data = pd.read_csv('train.csv')

import time
start = time.time()

loss = 0
h1nodes = 40
h2nodes = 20
batch_size = 128
iteration = 20
testtime = 1000
learning_rate = 0.001


Learning(iteration,learning_rate)


Test(testtime)
plot(iteration)


end = time.time()
print 'total time is ', (end - start)


