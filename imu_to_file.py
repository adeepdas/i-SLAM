import socket
import struct
import threading
import time
import numpy as np

IMU_PACKET_SIZE = 212
IMU_PORT = 13005
HOST = '0.0.0.0'

latest_imu_data = None
data_lock = threading.Lock()

def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))  # Connect to an external server
        return s.getsockname()[0]   # Get the local IP


def receive_imu_data(host, port):
    global latest_imu_data

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Starting IMU server on {host}:{port}, waiting for a connection...")
    connection, client_address = server_socket.accept()
    print(f"IMU Connection from {client_address}")

    try:
        while True:
            data_in = connection.recv(IMU_PACKET_SIZE)
            if not data_in:
                break
            
            if len(data_in) == IMU_PACKET_SIZE:
                timestamp, r_ax, r_ay, r_az, r_gx, r_gy, r_gz, r_mx, r_my, r_mz, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, yaw, pitch, roll, quat_x, quat_y, quat_z, quat_w = struct.unpack("dddddddddddddddddddddddddd", data_in[4:])
                imu_values = [
                    acc_x, acc_y, acc_z,  # Acc_X, Acc_Y, Acc_Z
                    gyro_x, gyro_y, gyro_z   # Gyro_X, Gyro_Y, Gyro_Z
                ]
                
                # timestamp = time.time()  # Get the current time in seconds (epoch time)
                
                with data_lock:
                    latest_imu_data = (timestamp, imu_values)

    except Exception as e:
        print(f"IMU Error: {e}")
    finally:
        connection.close()
        print("IMU Connection closed")

def write_imu_data_to_file(file_path):
    global latest_imu_data

    with open(file_path, "w") as f:
        f.write("timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z\n")  # CSV header
        while True:
            time.sleep(0.01)  # Write every 0.01 seconds
            with data_lock:
                if latest_imu_data is not None:
                    timestamp, imu_values = latest_imu_data
                    line = f"{timestamp}, " + ", ".join(f"{val:.6f}" for val in imu_values) + "\n"
                    f.write(line)
                    f.flush()  # Ensure real-time writing

if __name__ == "__main__":

    # get ip address
    ip_address = get_local_ip()
    print(f"Server IP address: {ip_address}")

    imu_thread = threading.Thread(target=receive_imu_data, args=(HOST, IMU_PORT))
    write_thread = threading.Thread(target=write_imu_data_to_file, args=("imu_data.csv",))

    imu_thread.start()
    write_thread.start()

    imu_thread.join()
    write_thread.join()
