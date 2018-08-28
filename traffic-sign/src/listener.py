#!/usr/bin/env python
#Author: Weilun Peng
#06/04/2018

import rospy
import numpy as np
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from traffic_sign_classifier import sign_predict
from traffic_sign_detector import sign_detect


class Image_regonize:

    def __init__(self, sg_det, sg_pred):
        self.bridge = CvBridge()
        self.sg_pred = sg_pred
        self.sg_det = sg_det
        self.image_sub = rospy.Subscriber('usb_cam/image_raw', Image, self.callback)
        
    def callback(self, data):
        try:
          cv_image_mat = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
        cv_image = np.asarray(cv_image_mat, dtype=np.uint8)
        print ("Start to detect")
        img_allsigns = self.sg_det.detect(cv_image)
        if not img_allsigns:
            print("No signs detect")
            return
        print ("Start to classify")
        all_result = []
        for each_sign in img_allsigns:
            resized_sign = cv2.resize(each_sign, (32, 32))
            Result_pred = self.sg_pred.predict(resized_sign)
            all_result.append(Result_pred)
        print (all_result)
        # cv2.imshow("Sub window", sub_img)
        # cv2.imshow("Full window", cv_image)
        # cv2.waitKey(1)
        # cv2.imshow("Image window", sub_img)
def main():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('image_classify', anonymous=True)

    sg_det = sign_detect()
    sg_pred = sign_predict()
    img_conv = Image_regonize(sg_det, sg_pred)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    main()
