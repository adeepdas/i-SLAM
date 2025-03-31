# Setup for Server Code on VM

This repository provides a Python binding for H.264 decoding using FFmpeg and pybind11.

## Installation Instructions

### **1. Configure Network in VirtualBox**

To ensure network connectivity in your VM, follow these steps:

1. Open **VirtualBox** settings.
2. Navigate to **Network** settings.
3. Set **Attached to:** `Bridged Adapter` to enable direct network access.

### **2. Install Required Dependencies**

Ensure your system has all necessary dependencies installed:

```sh
sudo apt update
sudo apt install -y cmake python3 python3-dev python3-pip \
                    libavcodec-dev libavutil-dev libswscale-dev \
                    build-essential
```

### **3. Clone the Repository**

Download and enter the repository:

```sh
git clone https://github.com/DaWelter/h264decoder.git
cd h264decoder
```

### **4. Build and Install the Project**

Use `pip` to install the package:

```sh
pip install setuptools wheel
pip install .
```

---

For additional details, refer to the [GitHub repository](https://github.com/DaWelter/h264decoder).

