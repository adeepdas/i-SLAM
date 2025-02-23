# i-SLAM
Code Repo for ROB 530 Final Project - Group 1. 

We are creating i-SLAM, a real-time “slap on” visual SLAM system that leverages an iPhone’s IMU sensors and depth camera. Instead of the standard SIFT+PnP pipeline, our approach uses the end-to-end SRPose framework for two-view relative pose estimation. This design allows you to simply attach your mobile device to any moving object in a static environment and track its location accurately. We fuse RGB, depth, and inertial data to handle various camera intrinsics and image sizes. Data is streamed to a server for computation, and the pose results are displayed in real-time through a web interface. The system is designed for easy setup, making it adaptable to numerous applications, especially in education.

