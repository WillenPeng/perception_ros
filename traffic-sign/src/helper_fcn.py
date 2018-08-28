# Load modules
import cv2
import csv
import time
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt


#Some useful image functions	
def eq_Hist(img):
	#Histogram Equalization
	img2=img.copy() 
	img2[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
	img2[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
	img2[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
	return img2

def scale_img(img):
	img_size = img.shape[0]
	img2=img.copy()
	sc_y=0.4*np.random.rand()+1.0
	img2=cv2.resize(img, None, fx=1, fy=sc_y, interpolation = cv2.INTER_CUBIC)
	c_x,c_y, sh = int(img2.shape[0]/2), int(img2.shape[1]/2), int(img_size/2)
	return img2

def crop(img, mar=0, ):
	img_size = img.shape[0]
	c_x,c_y, sh = int(img.shape[0]/2), int(img.shape[1]/2), int(img_size/2-mar)
	return img[(c_x-sh):(c_x+sh),(c_y-sh):(c_y+sh)]

def rotate_img(img):
	c_x,c_y = int(img.shape[0]/2), int(img.shape[1]/2)
	ang = 30.0*np.random.rand()-15
	Mat = cv2.getRotationMatrix2D((c_x, c_y), ang, 1.0)
	return cv2.warpAffine(img, Mat, img.shape[:2])

def sharpen_img(img):
	gb = cv2.GaussianBlur(img, (5,5), 20.0)
	return cv2.addWeighted(img, 2, gb, -1, 0)
#Compute linear image transformation ing*s+m
def lin_img(img,s=1.0,m=0.0):
	img2=cv2.multiply(img, np.array([s]))
	return cv2.add(img2, np.array([m]))

#Change image contrast; s>1 - increase
def contr_img(img, s=1.0):
	m=127.0*(1.0-s)
	return lin_img(img, s, m)

def transform_img(img):
	img2=sharpen_img(img)
	img2=crop(img2,3)
	img2=contr_img(img2, 1.5)
	return eq_Hist(img2)

def augment_img(img):
	img=contr_img(img, 1.8*np.random.rand()+0.2)
	img=rotate_img(img)
	img=scale_img(img)
	return transform_img(img)