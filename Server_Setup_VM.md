# Setup for Server Code on VM

This repository provides a Python binding for H.264 decoding using FFmpeg and pybind11.

## Installation Instructions

### **1. Install Dependencies**

Ensure you have the required dependencies installed:

```sh
sudo apt update
sudo apt install -y cmake python3 python3-dev python3-pip \
                    libavcodec-dev libavutil-dev libswscale-dev \
                    build-essential
```

### **2. Clone the Repository**

```sh
git clone https://github.com/DaWelter/h264decoder.git
cd h264decoder
```

### **3. Build the Project**

Use `pip` to install the package:

```sh
pip install setuptools wheel
pip install .
```

### 4. Network Config in Virtual Box

1. Go Virtual Box settings
2. Go to Network
   1. Set Attached to: Bridged Adapter

---

For additional details, refer to the [GitHub repository](https://github.com/DaWelter/h264decoder).

