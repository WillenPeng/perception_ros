1. roscore
2. open camera driver
rosrun usb_cam usb_cam_node _video_device:=/dev/video1
3. source under catkin_ws
source devel/setup.bash
4. rosrun
5. ##use command to get an image view (under catkin_ws)
rosrun image_view image_view image:=/usb_cam/image_raw
