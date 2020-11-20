#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 21:12:57 2020

@author: jiewang
"""

import numpy as np
import cv2
import os 
import glob
import pickle 

def main():

    img_dir = "/Users/jiewang/Documents/BIC Notes/MIT DATASET/TEST" # Enter Directory of all images  
    data_path = os.path.join(img_dir,'*g') 
    files = glob.glob(data_path) 
    data = [] 
    Features = []
    count = 0
    for f1 in files: 
        img = cv2.imread(f1, cv2.IMREAD_COLOR) 
        data.append(img)
        [matrix, F] = findDistToCenter(img)
        if count == 0:
            resize = matrix
        Features.append(F)
        count = count + 1
        print(count)
    #Features = np.matrix(Features)
    
    return Features, resize

def findDistToCenter(img):
    wd, ht, cc = img.shape
    midX = np.floor(wd/2)
    midY = np.floor(ht/2)

    
    distMatrix = np.zeros((wd,ht))
    for i in range(0,wd):
        for j in range(0, ht):
            distMatrix[i,j] = np.floor(np.sqrt((i-midX)**2+(j-midY)**2))
    
    distMatrix = distMatrix/max(np.reshape(distMatrix.transpose(),(wd*ht,1)))
    distMatrix = cv2.resize(distMatrix, (200,200), interpolation=cv2.INTER_LINEAR)
    features = np.reshape(distMatrix.transpose(),(200**2,1))
    
    return distMatrix, features

if __name__ == '__main__':
    Features,resize = main()
    f = open('store.pckl', 'wb')
    pickle.dump(Features, f)
    f.close()