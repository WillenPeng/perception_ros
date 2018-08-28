from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
import tensorflow as tf
import time
from optparse import OptionParser
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

config_output_filename = "/home//nvidia/catkin_ws/src/traffic-sign/src/train_model_detector/config.pickle"
MODEL_PATH = "/home//nvidia/catkin_ws/src/traffic-sign/src/train_model_detector/mobilenet_pluse_rec.hdf5"
bbox_threshold = 0.995
NUMROIS = 512
THRELENGTH = 32

class sign_detect:
	"""docstring for sign_detect"""
	def __init__(self):
		K.set_session(get_session(0.4))
		sys.setrecursionlimit(40000)

		with open(config_output_filename, 'rb') as f_in:
			self.C = pickle.load(f_in)

		if self.C.network == 'resnet50':
			import keras_frcnn.resnet as nn
		elif self.C.network == 'vgg':
			import keras_frcnn.vgg as nn
		elif self.C.network == 'mobilenet':
			import keras_frcnn.mobilenet as nn
		# turn off any data augmentation at test time
		self.C.use_horizontal_flips = False
		self.C.use_vertical_flips = False
		self.C.rot_90 = False

		class_mapping = self.C.class_mapping
		if 'bg' not in class_mapping:
			class_mapping['bg'] = len(class_mapping)
		class_mapping = {v: k for k, v in class_mapping.items()}
		self.class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
		self.class_mapping = class_mapping
		self.C.num_rois = NUMROIS #!!!!!!!!!!!!!

		if self.C.network == 'resnet50':
			num_features = 1024

		elif self.C.network == 'vgg':
			num_features = 512

		elif self.C.network == 'mobilenet':
			num_features = 512

		if K.image_dim_ordering() == 'th':
			input_shape_img = (3, None, None)
			input_shape_features = (num_features, None, None)
		else:
			input_shape_img = (None, None, 3)
			input_shape_features = (None, None, num_features)

		img_input = Input(shape=input_shape_img)
		roi_input = Input(shape=(self.C.num_rois, 4))
		feature_map_input = Input(shape=input_shape_features)

		# define the base network (resnet here, can be VGG, Inception, etc)
		shared_layers = nn.nn_base(img_input, trainable=True)

		# define the RPN, built on the base layers
		num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
		rpn_layers = nn.rpn(shared_layers, num_anchors)

		classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(class_mapping), trainable=True)

		self.model_rpn = Model(img_input, rpn_layers)
		self.model_classifier_only = Model([feature_map_input, roi_input], classifier)
		self.model_classifier = Model([feature_map_input, roi_input], classifier)

		self.model_rpn.load_weights(MODEL_PATH, by_name=True)
		self.model_classifier.load_weights(MODEL_PATH, by_name=True)
		self.model_rpn.compile(optimizer='sgd', loss='mse')
		self.model_classifier.compile(optimizer='sgd', loss='mse')
		self.graph = tf.get_default_graph()

		self.image_pub = rospy.Publisher("img_sign_detect",Image, queue_size=10)
		self.bridge = CvBridge()

	def detect(self, img):
		with self.graph.as_default():

			X, ratio = format_img(img, self.C)
			if K.image_dim_ordering() == 'tf':
				X = np.transpose(X, (0, 2, 3, 1))
			
			[Y1, Y2, F] = self.model_rpn.predict(X)
			R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), max_boxes=64, overlap_thresh=0.7)
			# convert from (x1,y1,x2,y2) to (x,y,w,h)
			R[:, 2] -= R[:, 0]
			R[:, 3] -= R[:, 1]

			bboxes = {}
			probs = {}
			for jk in range(R.shape[0] // self.C.num_rois + 1):
				ROIs = np.expand_dims(R[self.C.num_rois * jk:self.C.num_rois * (jk + 1), :], axis=0)
				if ROIs.shape[1] == 0:
					break
				if jk == R.shape[0] // self.C.num_rois:
					# pad R
					curr_shape = ROIs.shape
					target_shape = (curr_shape[0], self.C.num_rois, curr_shape[2])
					ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
					ROIs_padded[:, :curr_shape[1], :] = ROIs
					ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
					ROIs = ROIs_padded

				[P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

				for ii in range(P_cls.shape[1]):

					if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
						continue

					cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

					if cls_name not in bboxes:
						bboxes[cls_name] = []
						probs[cls_name] = []

					(x, y, w, h) = ROIs[0, ii, :]

					cls_num = np.argmax(P_cls[0, ii, :])
					try:
						(tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
						tx /= self.C.classifier_regr_std[0]
						ty /= self.C.classifier_regr_std[1]
						tw /= self.C.classifier_regr_std[2]
						th /= self.C.classifier_regr_std[3]
						x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)

					except:
						pass
					bboxes[cls_name].append(
						[self.C.rpn_stride * x, self.C.rpn_stride * y, self.C.rpn_stride * (x + w), self.C.rpn_stride * (y + h)])
					probs[cls_name].append(np.max(P_cls[0, ii, :]))
			all_signes = []      
			for key in bboxes:
				bbox = np.array(bboxes[key])

				new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.3)
				# todo : 0.3 is good
				for jk in range(new_boxes.shape[0]):
					(x1, y1, x2, y2) = new_boxes[jk, :]

					(real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

					if (real_x2 - real_x1) >= THRELENGTH and (real_y2 - real_y1) >= THRELENGTH:
						all_signes.append(img[real_x1:real_x2, real_y1:real_y2])
						# show image
						cv2.rectangle(img, (real_x1, real_y1), (real_x2, real_y2),
                          (int(self.class_to_color[key][0]), int(self.class_to_color[key][1]), int(self.class_to_color[key][2])), 2)
			# cv2.imshow("Full window", img)
   #      	cv2.waitKey(1)
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
		return all_signes


def get_session(gpu_fraction=0.3):
	"""
	This function is to allocate GPU memory a specific fraction
	Assume that you have 6GB of GPU memory and want to allocate ~2GB
	"""
	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def format_img_size(img, C):
	""" formats the image size based on config """
	img_min_side = float(C.im_size)
	(height, width, _) = img.shape

	if width <= height:
		ratio = img_min_side / width
		new_height = int(ratio * height)
		new_width = int(img_min_side)
	else:
		ratio = img_min_side / height
		new_width = int(ratio * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	return img, ratio

def format_img_channels(img, C):
	""" formats the image channels based on config """
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img

def format_img(img, C):
	""" formats an image for model prediction based on config """
	img, ratio = format_img_size(img, C)
	img = format_img_channels(img, C)
	return img, ratio

# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
	real_x1 = int(round(x1 // ratio))
	real_y1 = int(round(y1 // ratio))
	real_x2 = int(round(x2 // ratio))
	real_y2 = int(round(y2 // ratio))

	return (real_x1, real_y1, real_x2, real_y2)
