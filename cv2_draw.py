import cv2
import numpy as np
import os
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import logging
def read_image(IMG_DIR, IMG_NAME_1, IMG_NAME_2):
    img_1 = cv2.imread(IMG_DIR + IMG_NAME_1)
    img_2 = cv2.imread(IMG_DIR + IMG_NAME_2)
    #cv2.imshow(IMG_NAME_1,img_1)
    #cv2.imshow(IMG_NAME_2,img_2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img_1, img_2
IMG_DIR = 'image_0/'
logging.basicConfig(level=logging.INFO)
T = np.zeros((3, 1))
R = np.eye(3)
plt.ion()
fig, ax = plt.subplots()
plt.xlim(-100,200)
plt.ylim(-100,200)
X, Y, Z = [], [], []
sc = ax.scatter(X, Y)
plt.draw()
with open('KITTI  sample dataset/dataset/sequences/00/calib.txt', 'r') as f:
    P = re.split('\n| ', f.read())
p0 = np.array([float(x) for x in P[1:13]]).reshape((3, 4))
f1 = p0[0, 0]
pp = (p0[0, 2], p0[1, 2])
#print(f1)
#print(type(f1))
#print(pp)
#print(type(pp))
for root,dirs,files in os.walk(IMG_DIR):
    for i, f in enumerate(files):
        #print(i)
        #if i == 30:
            #print('here')
        #    break
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
        #cv2.imshow('image1',SIFT_1)
        #cv2.imshow('image2',SIFT_2)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        ###SIFT###
        #logging.info("SIFT is done.")
        ###FIND MATCHES###
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        good = []
        pts1 = []
        pts2 = []
        for j,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
        ###FIND MATCHES###
        #logging.info("Matching is done.")
        ###FIND ESSENTIALMATRIX###
        pts1_E = np.asarray(pts1)
        pts2_E = np.asarray(pts2)
        E, mask = cv2.findEssentialMat(pts1_E, pts2_E, focal = f1, pp = pp)
        pts1_E = pts1_E[mask.ravel()==1]
        pts2_E = pts2_E[mask.ravel()==1]
        points, r, t, mask = cv2.recoverPose(E, pts1_E, pts2_E)
        ###FIND ESSENTIALMATRIX###   
        #logging.info("Essential Matrix is found.")
        ###CALCULATE R AND T###
        R = np.linalg.inv(R)
        T += -R@t
        R = R@r
        ###CALCULATE R AND T###
        print(T.ravel())
        #if i>95 and i<110:
        #    print(T.ravel())
        #if i>195 and i<210:
        #    print(T.ravel())
        #logging.info("T and R are calculated.")
        ###PLOT###
        X.append(T[0, 0])
        Y.append(T[2, 0])
        #Z.append(T[2, 0])
        sc.set_offsets(np.c_[X, Y])
        #sp.set_data(X,Y)
        fig.canvas.draw_idle()
        plt.pause(0.1)
        ###PLOT###
        #print(T)
        IMG_NAME_1 = IMG_NAME_2
plt.waitforbuttonpress()