# Load modules
import cv2
import csv
import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from helper_fcn import transform_img
# TODO: Fill this in based on where you saved the training and testing data

class sign_predict:

	def __init__(self):
		signs_class=[]
		with open('/home//nvidia/catkin_ws/src/traffic-sign/src/train_model_classifier/signnames.csv', 'rt') as csvfile:
		    reader = csv.DictReader(csvfile, delimiter=',')
		    for row in reader:
		        signs_class.append((row['SignName']))
		self.signsclasses = signs_class

		saver = tf.train.import_meta_graph('/home//nvidia/catkin_ws/src/traffic-sign/src/train_model_classifier/my_net_f.ckpt.meta')
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		saver.restore(sess, '/home//nvidia/catkin_ws/src/traffic-sign/src/train_model_classifier/my_net_f.ckpt')
		graph = tf.get_default_graph()
		x = graph.get_tensor_by_name("x:0")
		keep_prob = graph.get_tensor_by_name("keep_prob:0")
		logits = graph.get_tensor_by_name("logits:0")
		
		self.x = x
		self.keep_prob = keep_prob
		self.logits = logits
		self.sess = sess

	def predict(self, img):
		# img = Image.open(test_img_paths)
		# img = np.array(img).astype('uint8')

		X_test_data=np.uint8(np.zeros((1,26,26,3)))
		X_test_data[0]=transform_img(img)
		X_test_data = X_test_data.reshape((-1, 26, 26, 3)).astype(np.float32)
		sign_class = self.sess.run(tf.argmax(self.logits, 1), feed_dict={self.x: X_test_data, self.keep_prob: 1.0})
		return self.signsclasses[sign_class[0]]

# def main():
#     # Parse arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('test_img_paths', help='Directory of images to predict')
#     args = parser.parse_args()

#     # Predict the image
#     signs_classes = predict(args.test_img_paths)
	
#     os._exit(0)

# if __name__ == '__main__':
#     main()
