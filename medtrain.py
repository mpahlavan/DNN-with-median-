# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 08:47:50 2019

@author: Mary
Python version: 3.6
"""

import numpy as np

class NeuralNetwork:
    def __init__(self, x, y, learning_rate, kernel=None):
        self.input = np.array(x)
        if kernel is None:
            self.kernel = np.random.rand(self.input.shape[0]-2,self.input.shape[1]-2)
        else:
            self.kernel = np.array(kernel)
        self.y = np.array(y)
        self.output = np.zeros(self.y.shape)
        self.output_index = np.zeros((self.y.shape[0],self.y.shape[1], 2),dtype=int)
        self.lr = learning_rate

    def feedforward(self):
        lenk2x = self.kernel.shape[0]//2
        lenk2y = self.kernel.shape[1]//2
        for i in range(lenk2x,self.input.shape[0]-lenk2x):
            for j in range(lenk2y,self.input.shape[1]-lenk2y):
                dotp = np.dot(self.input[i-lenk2x:i-lenk2x+self.kernel.shape[0],j-lenk2y:j-lenk2y+self.kernel.shape[1]], self.kernel)
                sin_dotp = np.sin(dotp)
                med = np.median(sin_dotp)
                med_index = np.where(sin_dotp == med)                
                self.output[i-lenk2x,j-lenk2y] = med
                self.output_index[i-lenk2x,j-lenk2y,:] = [med_index[0][0],med_index[1][0]]

    def backprop(self):
        dloss = 2*(self.y - self.output)
        lenk2x = self.kernel.shape[0]//2
        lenk2y = self.kernel.shape[1]//2
        for i in range(lenk2x,self.input.shape[0]-lenk2x):
            for j in range(lenk2y,self.input.shape[1]-lenk2y):
                idx = self.output_index[i-lenk2x,j-lenk2y]
                input_part = self.input[i-lenk2x:i-lenk2x+self.kernel.shape[0],j-lenk2y:j-lenk2y+self.kernel.shape[0]]
                input_element = input_part[idx[0],idx[1]]
                dy = -np.cos(self.kernel[idx[0],idx[1]]*input_element)
                dk = dloss[idx[0],idx[1]]*dy
                self.kernel[i-lenk2x,j-lenk2y] += self.lr * dk
    
    def loss(self):
        return np.sum(np.power(self.y - self.output,2))
    
    def learn(self, maxiteration=100, loss_thd=0.001):
        for iter in range(maxiteration):
            self.feedforward()
            if self.loss() < loss_thd:
                break
            self.backprop()
            print("iter ",iter, "loss", self.loss())
        
def main():
    I = [[1,2,3,4,5],
         [6,7,8,9,10],
         [11,12,13,14,15],
         [16,17,18,19,20],
         [21,22,23,24,25]]
    K = [[0.1,0.2,0.3],
         [0.4,0.5,0.6],
         [0.7,0.8,0.9]]
    T = [[0.5,0.5,0.5],
         [0.5,0.5,0.5],
         [0.5,0.5,0.5]]
    nn = NeuralNetwork(I,T,0.001,K)
    nn.learn(20000,0.0001)
    print(nn.kernel)
main()