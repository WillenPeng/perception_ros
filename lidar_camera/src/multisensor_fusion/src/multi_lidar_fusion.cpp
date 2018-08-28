#include "multi_lidar_fusion.h"


void constructFullClouds(pcl::PointCloud<pcl::PointXYZ>* scan,
                                    unsigned npoints, size_t &obs_count, pcl::PointCloud<pcl::PointXYZ>* obstacle_cloud)
{
  float min[grid_dim_][grid_dim_];
  float max[grid_dim_][grid_dim_];
  bool init[grid_dim_][grid_dim_];
  memset(&init, 0, grid_dim_*grid_dim_);
  
  // build height map
  for (unsigned i = 0; i < npoints; ++i) {
    int x = ((grid_dim_/2)+scan->points[i].x/m_per_cell_);
    int y = ((grid_dim_/2)+scan->points[i].y/m_per_cell_);
    if (x >= 0 && x < grid_dim_ && y >= 0 && y < grid_dim_) {
      if (!init[x][y]) {
        min[x][y] = scan->points[i].z;
        max[x][y] = scan->points[i].z;
        init[x][y] = true;
      } else {
        min[x][y] = MIN(min[x][y], scan->points[i].z);
        max[x][y] = MAX(max[x][y], scan->points[i].z);
      }
    }
  }

  // display points where map has height-difference > threshold
  for (unsigned i = 0; i < npoints; ++i) {
    int x = ((grid_dim_/2)+scan->points[i].x/m_per_cell_);
    int y = ((grid_dim_/2)+scan->points[i].y/m_per_cell_);
    if (x >= 0 && x < grid_dim_ && y >= 0 && y < grid_dim_ && init[x][y]) {
      if ((max[x][y] - min[x][y] > height_diff_threshold_) ) {   
        obstacle_cloud->points[obs_count].x = scan->points[i].x;
        obstacle_cloud->points[obs_count].y = scan->points[i].y;
        obstacle_cloud->points[obs_count].z = scan->points[i].z;
        //obstacle_cloud_.channels[0].values[obs_count] = (float) scan->points[i].intensity;
        obs_count++;
      } 
    }
  }
}


void Velo_pointcloud::callback_lidar_1(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptrCloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*cloud_msg,*ptrCloud);
    fprintf(f_1, "%f\n", cloud_msg->header.stamp.toSec());
    // ROS_INFO("Time stamp 1 %f", cloud_msg->header.stamp.toSec());
    // obstacle_cloud_.header.stamp = ptrCloud->header.stamp;
    // obstacle_cloud_.header.frame_id = ptrCloud->header.frame_id;

    // size_t npoints = ptrCloud->points.size();
    // obstacle_cloud_.points.resize(npoints);

    // size_t obs_count=0;


    // Velo_pointcloud::constructFullClouds(ptrCloud,npoints,obs_count);

    // obstacle_cloud_.points.resize(obs_count);

};


void Velo_pointcloud::callback_lidar_2(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptrCloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*cloud_msg,*ptrCloud);
    fprintf(f_2, "%f\n", cloud_msg->header.stamp.toSec());
    

};


void Velo_pointcloud::callback_lidar_3(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptrCloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::fromROSMsg(*cloud_msg,*ptrCloud);
    fprintf(f_3, "%f\n", cloud_msg->header.stamp.toSec());
   

};

void callback_lidar_sync(const sensor_msgs::PointCloud2ConstPtr& cloud_msg_1, 
                        const sensor_msgs::PointCloud2ConstPtr& cloud_msg_2, 
                        const sensor_msgs::PointCloud2ConstPtr& cloud_msg_3, 
                        const dbw_mkz_msgs::SteeringReportConstPtr& steering_report,
                        const sensor_msgs::ImuConstPtr& imu_report){
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptrCloud_1(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptrCloud_2(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ptrCloud_3(new pcl::PointCloud<pcl::PointXYZ>);


    fprintf(f_1, "%f, ", cloud_msg_1->header.stamp.toSec());
    fprintf(f_1, "%f, ", cloud_msg_2->header.stamp.toSec());
    fprintf(f_1, "%f, ", cloud_msg_3->header.stamp.toSec());
   
    pcl::fromROSMsg(*cloud_msg_1,*ptrCloud_1);
    pcl::fromROSMsg(*cloud_msg_2,*ptrCloud_2);
    pcl::fromROSMsg(*cloud_msg_3,*ptrCloud_3);

    // ******************************************************************
    // ******************** rigid body fuse ORIGINAL ********************
    // ******************************************************************
    velo_2.transform = Eigen::Matrix4f::Identity();
    velo_2.transform (0,3) = -0.0394873;
    velo_2.transform (1,3) = 0;
    velo_2.transform (2,3) = 0.0190585;
    velo_2.transform (0,1) = 0.0357845;
    velo_2.transform (1,0) = -0.0357854;

    pcl::transformPointCloud (*ptrCloud_2, velo_2.transformed_cloud, velo_2.transform);
    // // ROS_INFO("%f %f\n", dx_1, dx_2);


    velo_3.transform = Eigen::Matrix4f::Identity();
    velo_3.transform (0,3) = -0.0203914;
    velo_3.transform (1,3) = 0;
    velo_3.transform (2,3) = 0.0248587;
    pcl::transformPointCloud (*ptrCloud_3, velo_3.transformed_cloud, velo_3.transform);
    

    // pcl::PointCloud<pcl::PointXYZ> quarter;
    // quarter.header.stamp = ptrCloud_1->header.stamp;
    // quarter.header.frame_id = ptrCloud_1->header.frame_id;
    // for (int i = 0; i < ptrCloud_1->size()/4; ++i){
    //     quarter.push_back((*ptrCloud_1)[i]);
    //     quarter[i].x += steering_report->speed*i*0.1/ptrCloud_1->size();
    // }

    // cloud_fuse_wo_speed = quarter;


    cloud_fuse_wo_speed = velo_3.transformed_cloud;
    cloud_fuse_wo_speed += velo_2.transformed_cloud;
    cloud_fuse_wo_speed += *ptrCloud_1;

    obstacle_cloud_.clear();
    obstacle_cloud_.header.stamp = cloud_fuse_wo_speed.header.stamp;
    obstacle_cloud_.header.frame_id = cloud_fuse_wo_speed.header.frame_id;
    size_t npoints = cloud_fuse_wo_speed.points.size();
    obstacle_cloud_.points.resize(npoints);
    size_t obs_count=0;
    constructFullClouds(&cloud_fuse_wo_speed, npoints, obs_count, &obstacle_cloud_);


    sensor_msgs::PointCloud2 cloud_fuse_msg_wo_speed;
    pcl::toROSMsg(obstacle_cloud_,cloud_fuse_msg_wo_speed);
    pub_velo_wo_speed.publish(cloud_fuse_msg_wo_speed);

    // velo_1.cloud = *ptrCloud_1;
    // velo_2.cloud = *ptrCloud_2;
    // velo_3.cloud = *ptrCloud_3;



    // *********************************************************************
    // ********************** rotational compensation **********************
    // *********************************************************************
    
    for (int i = 0; i < ptrCloud_1->size(); ++i){  
        (*ptrCloud_1)[i].x += steering_report->speed*i*0.1/ptrCloud_1->size();
    }
    for (int i = 0; i < ptrCloud_2->size(); ++i){
        (*ptrCloud_2)[i].x += steering_report->speed*i*0.1/ptrCloud_2->size();
    }
    for (int i = 0; i < ptrCloud_3->size(); ++i){
        (*ptrCloud_3)[i].x += steering_report->speed*i*0.1/ptrCloud_3->size();
    }

    // *********************************************************************************
    // ********************* rigid body fuse with synchronization **********************
    // *********************************************************************************

    velo_2.transform = Eigen::Matrix4f::Identity();
    double dt_2 = cloud_msg_2->header.stamp.toSec() - cloud_msg_1->header.stamp.toSec();
    double dx_2 = dt_2*steering_report->speed;
    velo_2.transform(0,3) = -0.0394873+dx_2;
    velo_2.transform (1,3) = -0.418016;
    velo_2.transform(2,3) = 0.0190585;
    velo_2.transform(0,1) = 0.0357845;
    velo_2.transform(1,0) = -0.0357854;

    pcl::transformPointCloud (*ptrCloud_2, velo_2.transformed_cloud, velo_2.transform);

    velo_3.transform = Eigen::Matrix4f::Identity();
    double dt_3 = cloud_msg_3->header.stamp.toSec() - cloud_msg_1->header.stamp.toSec();
    double dx_3 = dt_3*steering_report->speed;
    
    velo_3.transform(0,3) = -0.0203914+dx_3;
    velo_3.transform (1,3) = 0.450944;
    velo_3.transform(2,3) = 0.0248587;
    pcl::transformPointCloud (*ptrCloud_3, velo_3.transformed_cloud, velo_3.transform);
    // ROS_INFO("%f %f\n", dx_2, dx_3);

    cloud_fuse = velo_3.transformed_cloud;
    cloud_fuse += velo_2.transformed_cloud;
    cloud_fuse += *ptrCloud_1;

    obstacle_cloud_.clear();
    obstacle_cloud_.header.stamp = cloud_fuse.header.stamp;
    obstacle_cloud_.header.frame_id = cloud_fuse.header.frame_id;
    npoints = cloud_fuse.points.size();
    obstacle_cloud_.points.resize(npoints);
    constructFullClouds(&cloud_fuse, npoints, obs_count, &obstacle_cloud_);

    sensor_msgs::PointCloud2 cloud_fuse_msg;
    pcl::toROSMsg(obstacle_cloud_,cloud_fuse_msg);

    pub_velo.publish(cloud_fuse_msg);

};

int
main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "velodyne_sub");
  ros::NodeHandle nh;

  grid_dim_ = 320;
  m_per_cell_ = 0.1;
  height_diff_threshold_ = 0.1;


  // ****************** synchronized ******************
  // open file to store the synchronized timestamp
  f_1 = fopen("time_1.txt","w");
  f_2 = fopen("time_2.txt","w");
  f_3 = fopen("time_3.txt","w");
  fprintf(f_1, "vlp1_t, vlp2_t, vlp3_t, imu_t, imu_omega_z, steering_t, speed\n");

  // Create a ROS subscriber for the input point cloud
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_1(nh, "/vlp16_1/velodyne_points", 1000);
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_2(nh, "/vlp16_2/velodyne_points", 1000);
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_3(nh, "/vlp16_3/velodyne_points", 1000);
  message_filters::Subscriber<dbw_mkz_msgs::SteeringReport> sub_v(nh, "/vehicle/steering_report", 1000);
  message_filters::Subscriber<sensor_msgs::Imu> sub_i(nh, "/novatel/imu", 1000);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, dbw_mkz_msgs::SteeringReport, sensor_msgs::Imu> MySyncPolicy;
  // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_1, sub_2, sub_3, sub_v, sub_i);
  sync.registerCallback(boost::bind(&callback_lidar_sync, _1, _2, _3, _4, _5));
  // message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_1, sub_2, sub_3);
  // sync.registerCallback(boost::bind(&callback_lidar_sync, _1, _2, _3));
  // ****************** unsynchronized ******************
  // f_1 = fopen("time_1_raw.txt","w");
  // f_2 = fopen("time_2_raw.txt","w");
  // f_3 = fopen("time_3_raw.txt","w");

  // ros::Subscriber sub_1 = nh.subscribe("/vlp16_1/velodyne_points", 1000, &Velo_pointcloud::callback_lidar_1, &velo_1, ros::TransportHints().tcpNoDelay(true));


  // ros::Subscriber sub_2 = nh.subscribe("/vlp16_2/velodyne_points", 1000, &Velo_pointcloud::callback_lidar_2, &velo_2, ros::TransportHints().tcpNoDelay(true));


  // ros::Subscriber sub_3 = nh.subscribe("/vlp16_3/velodyne_points", 1000, &Velo_pointcloud::callback_lidar_3, &velo_3, ros::TransportHints().tcpNoDelay(true));


  pub_velo = nh.advertise<sensor_msgs::PointCloud2>("/vlp_fuse", 1);
  pub_velo_wo_speed = nh.advertise<sensor_msgs::PointCloud2>("/vlp_fuse_wo_speed", 1);

  ros::spin();
  // Spin
}
