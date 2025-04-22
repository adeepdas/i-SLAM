# iLoco - Visual-Inertial SLAM System
Code Repository for ROB 530 Final Project - Group 1

## Overview
iLoco is a Visual-Inertial SLAM (Simultaneous Localization and Mapping) system that combines data from an iPhone's Lidar camera and IMU sensors to perform real-time trajectory estimation. The system consists of three main components:

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
- MacOs with Xcode for building and deploying the app
- iOS device with Lidar Camera (iPhone 12 Pro or later Pro models)
- iOS 13.0 or later

## Installation

### 1. System Dependencies (Ubuntu/Debian)
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

### 2. Miniconda Installation
```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Make installer executable
chmod +x Miniconda3-latest-Linux-x86_64.sh

# Run installer (follow the prompts)
./Miniconda3-latest-Linux-x86_64.sh

# Activate conda (or restart your terminal)
source ~/.bashrc
```

### 3. Create and Setup Conda Environment
```bash
conda create -n iLoco2 -y \
    python==3.9 \
    numpy==1.26.4 \
    conda-forge::gtsam==4.2.0 \
    conda-forge::matplotlib==3.8.3 \
    conda-forge::jupyterlab \
    conda-forge::opencv==4.8.0 \
    conda-forge::pillow \
    scipy \
    pip
```
4. Clone the repository:
```bash
git clone https://github.com/adeepdas/iLoco.git
cd iLoco
```

5. Build and install h264decoder and iLoco:
```bash
pip install . h264decoder/
```

### iOS Client Setup //TODO: CHANGE THIS
1. Open `VideoStreamingClient/VideoStreamingClient.xcodeproj` in Xcode
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
python video_imu_server.py

# Save files with custom name prefix
python video_imu_server.py --name experiment1
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

### Data Collection Duration
- By default, the server records for 60 seconds
- This can be modified by changing `SECONDS_TO_RECORD` in `video_imu_server.py`
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
python -m iLoco.gtsam_iter --imu data/imu_data.npy --video data/video_data.npy
```

## Project Structure
```
iLoco/
├── iLoco/                      # Main Python package
│   ├── imu_integration.py      # IMU processing
│   ├── visual_odometry.py      # Visual odometry
│   ├── gtsam_iter.py           # GTSAM optimization
│   └── visualization.py        # Trajectory visualization
├── h264decoder/                # H264 video decoder module
├── VideoStreamingClient/       # iOS client application
├── video_imu_server.py         # Data collection server
└── requirements.txt            # Python dependencies
```

## Data Format
The system saves data in two formats:

The system automatically saves video and IMU data to files after a predefined number of seconds. By default, this is set to 60 seconds. To change this duration, modify the `SECONDS_TO_RECORD` variable in the `video_imu_server.py` file.

| Data Type | File Format | Contents |
|-----------|-------------|----------|
| IMU       | `.npy`      | - Timestamp (float64) <br> - Accelerometer readings (float64, 3D vector: ax, ay, az) <br> - Gyroscope readings (float64, 3D vector: gx, gy, gz) |
| Video     | `.npy`      | - Timestamp (float64) <br> - Camera intrinsics (float32, fx, fy, cx, cy) <br> - RGB frames (uint8, 180x320x3) <br> - Depth data (float16, 180x320) <br>  |


## License
[Add your license information here]

## Contributors
Adeep Das, Velu Manohar, Nikhil Sridhar, Muhammad Khan
