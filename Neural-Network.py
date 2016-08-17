# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 02:15:09 2016

@author: izuki
"""
# ­*­ coding: utf­8 ­*­ """ Spyder Editor
import pandas as pd
import numpy as np

def sigmoid(vector):
    return 1 / (1 + np.exp(vector))

def sigmoidGradient(vector):
    return sigmoid(vector).T.A * (1 - sigmoid(vector)).T.A



#Data Acquisition
#data = pd.read_csv('train.csv')


i = 3  #ith row
pixels = data.iloc[i,1:]
label = data.iloc[i,0]
teacher = np.zeros(10)
teacher[label] = 1
print 'teacher = ', teacher



#Layer Setting
h1nodes = 20
h2nodes = 15
outputnum = 10
inputs = np.array(pixels,dtype = float)/255


#Thetas Initialization
theta1 = np.random.rand(len(inputs),h1nodes)
theta1 = np.matrix(theta1,float)
        
theta2 = np.random.rand(h1nodes,h2nodes)
theta2 = np.matrix(theta2,float)
        
theta3 = np.random.rand(h2nodes,outputnum)        
theta3 = np.matrix(theta3, float)

#Delta Theta
D1 = np.zeros((theta1.shape))
D2 = np.zeros((theta2.shape))
D3 = np.zeros((theta3.shape))

def Forward_Propagation (a1):
    #Layer1
    global z1, z2, z3
    global a2, a3, a4
    
    z2 = np.dot(a1.T,theta1)/100
    
    #Layer2
    a2 = sigmoid(z2)
    z3 = np.dot(a2,theta2)
    
    #Layer3    
    a3 = sigmoid(z3)
    z4 = np.dot(a3, theta3)
    
    #Output
    a4 = sigmoid(z4)

    return a4

def Back_Propagation(a4,teacher):
    
    global del4,del3,del2,del1
    
    print 'Back Propagation'
    #delta calculation    
    del4 = (teacher - a4).T
    print 'del4 =', del4
    
    del3 = (theta3*del4).A * sigmoidGradient(z3)
    print 'del3 =', del3
    
    del2 = (theta2*del3).A * sigmoidGradient(z2)
    print 'del2 =', del2
    
    print 'No del1'
    
    #delta 
    global D1, D2, D3



    D3 += np.dot(a3.T, del4.T)
    print 'D3 = ', D3
    D2 += np.dot(a2.T, del3.T)
    print 'D2 = ', D2
    D1 += np.dot(inputs.T, del2.T)
    print 'D1 = ', D1
    #thetas update
    
outputs = Forward_Propagation (inputs)
print outputs
Back_Propagation(outputs, teacher)