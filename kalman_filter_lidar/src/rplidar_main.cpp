// Weilun Peng
// 06/11/2018

#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/PointCloud.h"
#include "Eigen/Dense"
#include "FusionEKF.h"
#include "ukf.h"
#include "measurement_package.h"
#include <sstream>
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

#define RAD2DEG(x) ((x)*180./M_PI)

class SubscribeAndPublish
{
public:
  SubscribeAndPublish(int kal_num)
  {
    //Topic you want to publish
    if (kal_num == 1 || kal_num == 2) {
      kal_num_ = kal_num;
      pub_ = n_.advertise<sensor_msgs::PointCloud>("/filtered_cloud", 1000);

      //Topic you want to subscribe
      sub_ = n_.subscribe<sensor_msgs::LaserScan>("/scan", 1, &SubscribeAndPublish::scanCallback, this);
    }
    else {
      cout << "ERROR: Input of the Choice of Kalman Filter (1: EKF, 2:UKF)" <<endl;
      exit(-1);
    }
    
  }

  void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan)
  {
    int count = scan->scan_time / scan->time_increment;
    ROS_INFO("I heard a laser scan %s[%d]:", scan->header.frame_id.c_str(), count);
    ROS_INFO("angle_range, %f, %f", RAD2DEG(scan->angle_min), RAD2DEG(scan->angle_max));
    sensor_msgs::PointCloud filtered_cloud;
    filtered_cloud.header.frame_id = "laser";
    if (kal_num_ == 1) {
      // Create a EKF instance
      FusionEKF fusionEKF;
      fusionEKF_ = fusionEKF;
    }
    else {
      // Create a UKF instance
      UKF ukf;
      ukf_ = ukf;
    }
    
    for(int i = 0; i < count; i++) {
        geometry_msgs::Point32 filtered_point;

        float rad = scan->angle_min + scan->angle_increment * i;
        float cur_ranges = scan->ranges[i];
        float x = cur_ranges*cos(rad);
        float y = cur_ranges*sin(rad);
        if (cur_ranges > scan->range_max) {
            cur_ranges = scan->range_max;
            filtered_point.x = cur_ranges*cos(rad);
            filtered_point.y = cur_ranges*sin(rad);
            filtered_point.z = 0;
        }
        else if (cur_ranges < scan->range_min) {
            cur_ranges = scan->range_min;
            filtered_point.x = cur_ranges*cos(rad);
            filtered_point.y = cur_ranges*sin(rad);
            filtered_point.z = 0;
        }
        else {
            MeasurementPackage meas_package;
            meas_package.sensor_type_ = MeasurementPackage::LASER;
            meas_package.raw_measurements_ = VectorXd(2);
            meas_package.raw_measurements_ << x, y;
            meas_package.timestamp_ = (long long) (i*scan->time_increment*1000000);
            if (kal_num_ == 1) {
              fusionEKF_.ProcessMeasurement(meas_package);
              filtered_point.x = fusionEKF_.ekf_.x_(0);
              filtered_point.y = fusionEKF_.ekf_.x_(1);
              filtered_point.z = 0;
            }
            else {
              ukf_.ProcessMeasurement(meas_package);
              filtered_point.x = ukf_.x_(0);
              filtered_point.y = ukf_.x_(1);
              filtered_point.z = 0;
            }
        }
        // if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
          // output the estimation
        filtered_cloud.points.push_back(filtered_point);
        cout << " x_o: " << x << " x_f: " << filtered_point.x << " y_o: " << y  << " y_f: " << filtered_point.y << endl;
    }
    pub_.publish(filtered_cloud);
  }

private:
  int kal_num_;
  ros::NodeHandle n_; 
  ros::Publisher pub_;
  ros::Subscriber sub_;
  FusionEKF fusionEKF_;
  UKF ukf_;
};//End of class SubscribeAndPublish

int main(int argc, char **argv)
{
  //Initiate ROS
  ros::init(argc, argv, "rplidar_main");

  if (argc != 2) {
    cout << "ERROR: # of Input" <<endl;
    exit(-1);
  }
  //Create an object of class SubscribeAndPublish that will take care of everything
  int kal_num = atoi(argv[1]);
  SubscribeAndPublish SAPrplidar(kal_num);

  ros::spin();

  return 0;
}
