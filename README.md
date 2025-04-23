# iLoco - Visual-Inertial SLAM System
Code Repository for ROB 530 Final Project - Group 1

## Overview
In this work, we present iLoco, a real-time, plug-and-play visual SLAM system that leverages the iPhone’s built-in RGB-D camera and IMU to enable accurate localization. By integrating a sensor suite from the iPhone, iLoco delivers robust pose estimation by utilizing ORB feature matching for RGB-D visual feature extraction and tracking, while GTSAM is employed to tightly integrate inertial measurements with visual odometry for enhanced robustness and accuracy. The system is engineered to be a “slap on” solution, requiring minimal setup and no external calibration, making it especially suitable for rapid prototyping, educational demonstrations, and accessible SLAM research. iLoco’s design prioritizes ease of use and adaptability, enabling a wide range of users—from students to developers—to harness the power of real-time SLAM using everyday mobile devices.

## System Components
1. iOS Client App (VideoStreamingClient)
2. Server for Data Collection
3. SLAM Pipeline

## System Requirements

### Server Requirements
- Ubuntu 18.04 or later 
- Python 3.x
- CMake
- Required Python packages (instructions below):


### iOS Client Requirements
- MacOS with XCode for building and deploying the app
- iOS device with Lidar Camera (iPhone 12 Pro or later Pro models)
- iOS 13.0 or later

## Installation

### 1. Configure Network in VirtualBox (if using VM)
1. Open **VirtualBox** settings
2. Navigate to **Network** settings
3. Set **Attached to:** `Bridged Adapter` to enable direct network access


### 2. System Dependencies (Ubuntu/Debian)
```bash
# Update package list and install dependencies
sudo apt update && sudo apt install -y \
    cmake \
    python3 \
    python3-dev \
    python3-pip \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    build-essential \
    wget \
    git
```

### 3. Anaconda Installation
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Make installer executable
chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh

# Run installer (follow the prompts)
./Anaconda3-2024.02-1-Linux-x86_64.sh

# Activate conda (or restart your terminal)
source ~/.bashrc
```

### 4. Clone Repository

```bash
git clone https://github.com/adeepdas/iLoco.git
cd iLoco
```


### 5. Or install through pip
```bash
conda create -n iLoco -y python==3.9
conda activate iLoco
pip install numpy==1.26.4 \
            gtsam==4.2.0 \
            matplotlib==3.8.3 \
            opencv-python==4.8.0 \
            pillow \
            scipy
```

### 6. Activate Conda Environment
```bash
conda activate iLoco
```
### 7. Build and Install Dependencies
Build and install h264decoder and iLoco:
```bash
pip install . server/h264decoder/
```

### iOS Client Setup 
1. Open `iphone-depth-streaming/LiDARDepth.xcodeproj` in XCode
2. Configure signing and team settings
3. Build and deploy to your iOS device

## Usage


### Network Requirements
- The iOS device and server must be connected to the same WiFi network
- For optimal performance:
  - Use a 5GHz WiFi network when possible
  - Minimize network traffic from other devices
  - Keep the iOS device and server physically close to the WiFi router
  - Lower network traffic will result in reduced latency
  - Avoid crowded WiFi channels
  - Consider setting up a dedicated WiFi network for the system

### Starting the Server
1. Run the server script with one of these options:

```bash
# Basic usage - files will be saved with timestamp
python server/video_imu_server.py

# Save files with custom name prefix
python server/video_imu_server.py --name experiment1
```

This will create:
- Without --name: 
  - `video_data_20231125_143022.npy`
  - `imu_data_20231125_143022.npy`
- With --name:
  - `video_data_experiment1.npy`
  - `imu_data_experiment1.npy`

2. The server will display:
   - Its IP address (needed for the iOS client)
   - Recording progress (frames collected)
   - Connection status
   - Data saving confirmation

3. If you would like to test our code with the data that we collected it is liked below:
  - https://drive.google.com/drive/folders/1IST5OF5nrwEODLp1NMbyN7sl_zUXKcXj?usp=drive_link
  

### Data Collection Duration
- By default, the server records for 60 seconds
- This can be modified by changing `SECONDS_TO_RECORD` in `server/video_imu_server.py`
- The server automatically saves data after the specified duration
- Recording progress is shown in the terminal:
  ```
  VIDEO frame 145/1800
  IMU frame 478/6000
  ```

### Running the iOS Client
1. Launch the app on your iOS device
2. Enter the server's IP address
3. Tap "Start" to begin streaming camera and IMU data

### Running Multiple Sessions
1. After each recording session, quit both the app and server
2. Relaunch both the app and server before starting a new session
3. This reset ensures proper initialization and data handling for each new recording

### Processing Data
The system provides several scripts for data processing and visualization:
- `imu_integration.py`: Process raw IMU data
- `visual_odometry.py`: Extract visual odometry from video frames
- `gtsam_iter.py`: Perform optimization using GTSAM

Example usage:
```bash
python -m algo.gtsam_iter --imu server/output/imu_data.npy --video server/output/video_data.npy
```

## Project Structure
```
iLoco/
├── server/                     # Server components
│   ├── h264decoder/           # H264 video decoder module
│   ├── video_imu_server.py    # Data collection server
│   └── output/                # Directory for recorded data
│       └── *.npy             # Recorded video and IMU data
├── iphone-depth-streaming/    # iOS client application
│   └── LiDARDepth.xcodeproj  # Xcode project
├── algo/                      # SLAM algorithms
│   ├── imu_integration.py    # IMU processing
│   ├── visual_odometry.py    # Visual odometry
│   ├── gtsam_iter.py        # GTSAM optimization
│   └── visualization.py      # Trajectory visualization
└── requirements.txt          # Python dependencies
```

## Data Storage
- All recorded data (`.npy` files) are stored in `server/output/` directory
- This directory is git-ignored to prevent large files from being committed
- Make sure to back up your data separately if needed

## Running the Server
Update the server execution command to reflect new organization:
```bash
# From the iLoco directory
python server/video_imu_server.py

# With custom name
python server/video_imu_server.py --name experiment1
```

## Processing Data
Update the processing commands to reflect new organization:
```bash
# From the iLoco directory
python algo/gtsam_iter.py --imu server/output/imu_data.npy --video server/output/video_data.npy
```

## Data Format
The system saves data in two formats:

The system automatically saves video and IMU data to files after a predefined number of seconds. By default, this is set to 60 seconds. To change this duration, modify the `SECONDS_TO_RECORD` variable in the `server/video_imu_server.py` file.

| Data Type | File Format | Contents |
|-----------|-------------|----------|
| IMU       | `.npy`      | - Timestamp (float64) <br> - Accelerometer readings (float64, 3D vector: ax, ay, az) <br> - Gyroscope readings (float64, 3D vector: gx, gy, gz) |
| Video     | `.npy`      | - Timestamp (float64) <br> - Camera intrinsics (float32, fx, fy, cx, cy) <br> - RGB frames (uint8, 180x320x3) <br> - Depth data (float16, 180x320) <br>  |


## License
This project is fully open-source. You can find the code, documentation, and setup instructions on our GitHub repository: [\[GitHub link\]](https://github.com/adeepdas/iLoco)

## Contributors
Adeep Das, Velu Manohar, Nikhil Sridhar, Muhammad Khan