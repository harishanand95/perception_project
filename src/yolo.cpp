/*
  Harish Anand
  04-18-2020
*/

#include "ros/ros.h"
#include "darknet_ros_msgs/BoundingBoxes.h"
#include "darknet_ros_msgs/BoundingBox.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <math.h>

// Alex: 
#include <group_1_project_3a/detection.h>

using namespace std;
using namespace cv;

// global variables
int xMin = 0;
int yMin = 0;
int xMax = 0;
int yMax = 0;
cv::Mat maskJeansHuman;
image_transport::Publisher pub;

// Alex:
ros::Publisher humanDetection;

void yoloCallback(const darknet_ros_msgs::BoundingBoxes::ConstPtr& msg)
{ 
    if (msg->bounding_boxes[0].Class == "person") {
        
        xMin = msg->bounding_boxes[0].xmin;
        // If bounding box goes beyong image size, 
        // set box corners to image size
        if (xMin <=0 )
          xMin = 0;
        else if (xMin >=640)
          xMin = 640;

        yMin = msg->bounding_boxes[0].ymin;
        if (yMin <=0 )
          yMin = 0;
        else if (yMin >=480)
          yMin = 480;  

        xMax = msg->bounding_boxes[0].xmax;
        if (xMax <=0 )
          xMax = 0;
        else if (xMax >=640)
          xMax = 640;

        yMax = msg->bounding_boxes[0].ymax;
        if (yMax <=0 )
          yMax = 0;
        else if (yMax >=480)
          yMax = 480;  
    }
}


void rgbCallback(const sensor_msgs::ImageConstPtr& original_image)
{

    cv_bridge::CvImagePtr cv_ptr;
    // Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
    try
    {
        cv_ptr = cv_bridge::toCvCopy(original_image);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    /*
    // code to write an ros image to file to do tests
    cv::Mat rgb_img = cv_ptr->image;
    string path = "/root/";
    string frame = "rgb_frame";
    string num = to_string(b);
    string new_num = std::string(4 - num.length(), '0') + num;
    string filename = path + frame + new_num + ".jpg";
    b = b + 1;
    cvtColor(rgb_img, rgb_img, CV_BGR2RGB);
    imwrite(filename, rgb_img);
    */
    cv::Mat3b rgbImg = cv_ptr->image;
    cv::Mat3b hsvImg;
    cvtColor(rgbImg, hsvImg, CV_RGB2HSV);
    inRange(hsvImg, Scalar(100, 150, 0), Scalar(140, 255, 255), maskJeansHuman);

}


void depthToCV8UC1(const cv::Mat& float_img, cv::Mat& mono8_img){
    // Process depth images
    if(mono8_img.rows != float_img.rows || mono8_img.cols != float_img.cols){
        mono8_img = cv::Mat(float_img.size(), CV_8UC1);}
    cv::convertScaleAbs(float_img, mono8_img, 3, 0.0);
}


void depthCallback(const sensor_msgs::ImageConstPtr& msg){
    
    cv_bridge::CvImagePtr cv_ptr;
    // Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // Convert depth image to 8bit unsigned with 1 channel
    cv::Mat depthImg = cv_ptr->image;
    cv::Mat depthImg8UC1;
    depthToCV8UC1(depthImg, depthImg8UC1);
    
    // A zeros mask where none of the pixels are activated
    cv::Mat mask = cv::Mat::zeros(depthImg8UC1.size(), depthImg8UC1.type()); 
    
    // TODO: mask NaN values 
    
    // Mask bounding box 
    cv::Point pt1(xMin, yMin);
    cv::Point pt2(xMax, yMax);
    mask(Rect(pt1, pt2)) = 255;

    // Mask the jeans of the human
    cv::bitwise_and(mask, maskJeansHuman, mask);

    // Mean of the depth values
    cv::Scalar_<double> mean = cv::mean(depthImg, mask);
    // ROS_INFO("%f", mean[0]);

    // No detections case
    group_1_project_3a::detection humanMsg;
    if (mean[0] <= 0.01f || isnan(mean[0])) {
      humanMsg.status = "false";
      humanMsg.x = 0;
      humanMsg.y = 0;
      humanMsg.z = 0;
      humanMsg.angle = 0;
      humanDetection.publish(humanMsg);
      return;
    }

    // Display mask
    sensor_msgs::ImagePtr maskMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask).toImageMsg();
    pub.publish(maskMsg);

    float u = ((xMax-xMin)/ 2 + xMin);
    float v = ((yMax-yMin)/ 2 + yMin);


    float x = ((u - 320.5) * mean[0]) / 565.6008952774197; // X/Z = ((U-Cx))/fx
    float y = ((v - 240.5) * mean[0]) / 565.6008952774197; // Y/Z = ((V-Cy))/fy
    float angle = asin ( x / mean[0]);

    ROS_INFO("X %f, Y %f, Z %f ", x, y, mean[0]);
    ROS_INFO("angle %f ", angle);

    // Alex:
    humanMsg.status = "true";
    humanMsg.x = x;
    humanMsg.y = y;
    humanMsg.z = mean[0];
    humanMsg.angle = angle;
    humanDetection.publish(humanMsg);

    // code to write an ros image to file to do tests
    // cv::Mat depth_float_img = cv_ptr->image;
    // cv::Mat depth_mono8_img;
    // string path = "/root/";
    // string frame = "depth_frame";
    // string num = to_string(a);
    // string new_num = std::string(4 - num.length(), '0') + num;
    // string filename = path + frame + new_num + ".jpg";
    // a = a + 1;
    // depthToCV8UC1(depth_float_img, depth_mono8_img);
    // imwrite(filename, depth_mono8_img);
}


void test(){
  string path = "/root/";
  string depthFrame = "depth_frame0001.jpg";
  string rgbFrame = "rgb_frame0001.jpg";
  cv::Mat3b depthImg = imread(path + depthFrame, CV_LOAD_IMAGE_COLOR); 
  cv::Mat3b bgrImg = imread(path + rgbFrame, CV_LOAD_IMAGE_COLOR); 

  // Apply color filter
  Mat3b hsvImg;
  cvtColor(bgrImg, hsvImg, CV_BGR2HSV);
  Mat mask;
  inRange(hsvImg, Scalar(100, 150, 0), Scalar(140, 255, 255), mask);
  cv::Scalar_<double> mean = cv::mean(depthImg, mask);
  cout<< mean;
  imshow( "Display window", mask); 
  waitKey(0);     
  
}



int main(int argc, char **argv)
{
  cout << argv[3];
  cout << argc;
  return 0;
  ros::init(argc, argv, "listener");
  //cout << argv[0];

  ros::NodeHandle n;
  srand(time(NULL)); // generate seed for random
  image_transport::ImageTransport it(n);
  pub = it.advertise("/mask", 1);
  // test();

  // Alex: 
  humanDetection = n.advertise<group_1_project_3a::detection>("/humanDetection", 1);

  ros::Subscriber yoloSubscriber = n.subscribe("/darknet_ros/bounding_boxes", 10, yoloCallback);
  ros::Subscriber depthSubscriber = n.subscribe("/tb3_0/camera/depth/image_raw", 10, depthCallback);
  ros::Subscriber rgbSubscriber = n.subscribe("/tb3_0/camera/rgb/image_raw", 10, rgbCallback);
  ros::spin();
  return 0;
}
