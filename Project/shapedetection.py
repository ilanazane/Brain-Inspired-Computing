#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:26:22 2020

@author: jiewang
"""
import numpy as np
import cv2
import os 
import glob



def main():

    img_dir = "/Users/jiewang/Documents/BIC Notes/MIT DATASET/TEST" # Enter Directory of all images  
    data_path = os.path.join(img_dir,'*g') 
    files = glob.glob(data_path) 
    #data = [] 


    for f1 in files: 
    #img = cv2.imread("/Users/jiewang/Documents/BIC Notes/MIT DATASET/TEST/i102423191.jpeg" , cv2.IMREAD_COLOR) 
        img = cv2.imread(f1, cv2.IMREAD_COLOR) 
        #data.append(img)
        findShape(img)
        
    #Features = np.matrix(Features)
    

    
def findShape(img):
    cimg = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img.copy()
    img = cv2.GaussianBlur(img,(7,7),1)
    img = cv2.Canny(img,127,255)
    th_img = img.copy()
    kernel = np.ones((5,5))
    img = cv2.dilate(img, kernel, iterations=1)
    cv2.imshow("dilated",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    circles = cv2.HoughCircles(image=img, method=cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50,param2=30,minRadius=1,maxRadius=70)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]:
        # draw the outer circle in green
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle in red
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    #cimg = cv2.resize(cimg,(200,200))
    cv2.imshow("Result",cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
   main()

    