# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:48:05 2019

@author: Yu Chen
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
import re

def read_image(IMG_DIR, IMG_NAME_1, IMG_NAME_2):
    img_1 = cv2.imread(IMG_DIR + IMG_NAME_1)
    img_2 = cv2.imread(IMG_DIR + IMG_NAME_2)
    return img_1, img_2

def addData(data,buf):
    buf = np.roll(buf,1)
    buf[:,0] = data
    return buf

#def real_plot(xx_data,yy_data,mark,xx_truth,yy_truth,pause_time=0.001):
#    if mark==[]:
#        mark, = ax.plot(xx_data,yy_data,'ro',xx_truth,yy_truth,'b^')
#    mark.set_data(xx_data,yy_data,xx_truth,yy_truth)
#    #plt.draw()
#    plt.pause(pause_time)
#    return mark

def real_plot(xx_data,yy_data,mark,pause_time=0.001):
    if mark==[]:
        mark, = ax1.plot(xx_data,yy_data,'o',alpha=0.8)
    mark.set_data(xx_data,yy_data)
    #plt.draw()
    plt.pause(pause_time)
    return mark

def true_plot(xx_data,yy_data,true,pause_time=0.001):
    if true==[]:
        true, = ax2.plot(xx_data,yy_data,'o',alpha=0.8)
    true.set_data(xx_data,yy_data)
    #plt.draw()
    plt.pause(pause_time)
    return true


IMG_DIR = 'image_0/'
logging.basicConfig(level=logging.INFO)

with open('KITTI  sample dataset/dataset/poses/00.txt', 'r') as f:
    transformation = np.array([float(x) if len(x) else None for x in re.split('\n| ', f.read())])
size = len(transformation) // 12
translation = np.reshape(transformation[3::4], (size, 3))

T = np.zeros((3, 1))
R = np.eye(3)


# Create figure for plotting
plt.close('all')
G = gridspec.GridSpec(1, 2)
#plt.ion()

fig1, ax1 = plt.subplots()
plt.style.use('ggplot')
plt.xticks(())
plt.yticks(())
plt.ylim([-100,100])
plt.xlim([-100,100])
plt.ylabel('Backward <--> Forward', fontsize='12')
plt.xlabel('Left  <-->  Right', fontsize='12')


fig2, ax2 =plt.subplots()
plt.style.use('ggplot')
plt.xticks(())
plt.yticks(())
plt.ylim([-400,400])
plt.xlim([-400,400])
plt.ylabel('Backward <--> Forward', fontsize='12')
plt.xlabel('Left  <-->  Right', fontsize='12')



mark = []
true = []
memory = 500
buffer1 = np.zeros([2,memory])
buffer2 = np.zeros([2,memory])


for root,dirs,files in os.walk(IMG_DIR):
    for i, f in enumerate(files):
        IMG_NAME_2 = f
        if f == '000000.png':
            IMG_NAME_1 = IMG_NAME_2
            continue
        img_1, img_2 = read_image(IMG_DIR, IMG_NAME_1, IMG_NAME_2)
        #logging.info(f + "is read.")
        #print(f + "is read.")
        
        ###SIFT###
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img_1,None)
        kp2, des2 = sift.detectAndCompute(img_2,None)
        SIFT_1 = cv2.drawKeypoints(img_1, kp1, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        SIFT_2 = cv2.drawKeypoints(img_2, kp2, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        ###FIND MATCHES###
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        pts1 = []
        pts2 = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                
        ###FIND ESSENTIALMATRIX###
        pts1_E = np.asarray(pts1)
        pts2_E = np.asarray(pts2)
        E, mask = cv2.findEssentialMat(pts1_E, pts2_E)
        pts1_E = pts1_E[mask.ravel()==1]
        pts2_E = pts2_E[mask.ravel()==1]
        points, r, t, mask = cv2.recoverPose(E, pts1_E, pts2_E)
        

        ###CALCULATE R AND T###
        T += -R@t
        R = r@R
        
        ###PLOT###
        
        DATA = np.zeros(2)
        DATA[0] = T[0, 0]
        DATA[1] = T[1, 0]
        
        DATA2 = np.zeros(2)
        DATA2[0] = translation[i, 0]
        DATA2[1] = translation[i, 1]

        buffer1 = addData(DATA,buffer1)
        buffer2 = addData(DATA2,buffer2)
        
        Y = buffer1[0,:]
        X = -buffer1[1,:]
        
        Yt = buffer2[0,:]
        Xt = buffer2[1,:]



        mark = real_plot(X,Y,mark)
        true = true_plot(Xt,Yt,true)
        
#        if (i >= 400):
#            break
        
        
        ###PLOT###
        #print(T)
        IMG_NAME_1 = IMG_NAME_2
        