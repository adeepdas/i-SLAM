import socket
import struct
import numpy as np
import cv2
import h264decoder  # Import the decoder from the cloned repository
import threading
import queue
from datetime import datetime

START_MARKER = b'\xAB\xCD\xEF\x01'  # 4-byte unique identifier
VIDEO_HEADER_SIZE = 24  # 4 (marker) + 8 (timestamp) + 4 (video size) + 4 (depth size) + 4 (intrinsic size)
IMU_PACKET_SIZE =  60  # 4 header + 8 timestamp + 8 accX + 8 accY + 8 accZ + 8 gyroX + 8 gyroY + 8 gyroZ 

# Server Config
VIDEO_PORT = 25005
IMU_PORT = 13005

DEPTH_HEIGHT = 180
DEPTH_WIDTH = 320

HOST = '0.0.0.0'  # Listen on all interfaces

SEONDS_TO_RECORD = 15
IMU_FRAME_RATE = 100
VIDEO_FRAME_RATE = 30

DEBUG_FRAME_COUNTER_VIDEO = SEONDS_TO_RECORD * VIDEO_FRAME_RATE
DEBUG_FRAME_COUNTER_IMU = SEONDS_TO_RECORD * IMU_FRAME_RATE 


video_frame_queue = queue.Queue(maxsize=10)
depth_frame_queue = queue.Queue(maxsize=10)

# --- Global variables for recording frames ---
# Define a structured dtype for each frame
video_frame = np.dtype([
    ('timestamp', np.float64),
    ('fx', np.float32),
    ('fy', np.float32),
    ('cx', np.float32),
    ('cy', np.float32),
    ('bgr', np.uint8, (180, 320, 3)),      # 320x180 BGR image
    ('depth', np.float16, (180, 320))      # 320x180 depth image in raw float16
])

imu_frame = np.dtype([
    ('timestamp', np.float64),
    ('acc', np.float64, (3,)),    # 3D accelerometer data
    ('gyro', np.float64, (3,)),   # 3D gyroscope data
])


# Pre-allocate an array for 10 frames
recorded_frames_video = np.empty(DEBUG_FRAME_COUNTER_VIDEO, dtype=video_frame)
recorded_frames_imu = np.empty(DEBUG_FRAME_COUNTER_IMU, dtype=imu_frame)

recorded_count_video = 0
recorded_count_imu = 0

file_written_video = False  # To ensure file write happens only once
file_written_imu = False  # To ensure file write happens only once

def write_video_frames_to_file():
    # Save the full structured array as a .npy
    filename = f"video_data_{datetime.now().strftime('%Y%m%d_%H%M')}.npy"
    np.save(filename, recorded_frames_video)

def write_imu_frames_to_file():
    # Save the full structured array as a .npy
    filename = f"imu_data_{datetime.now().strftime('%Y%m%d_%H%M')}.npy"
    np.save(filename, recorded_frames_imu)

def get_local_ip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("8.8.8.8", 80))  # Connect to an external server
        return s.getsockname()[0]   # Get the local IP

def extract_packet(buffer):
    """
    Extracts a full packet from the buffer if the start marker is detected.
    Returns the extracted packet and the remaining buffer.
    """
    marker_index = buffer.find(START_MARKER)
    if marker_index == -1:
        return None, buffer  # No start marker found

    if len(buffer) < marker_index + VIDEO_HEADER_SIZE:
        return None, buffer  # Not enough data for a complete header

    # Extract header information
    header = buffer[marker_index:marker_index + VIDEO_HEADER_SIZE]
    timestamp, video_size, depth_size, intrinsic_size = struct.unpack("dIII", header[4:VIDEO_HEADER_SIZE])
    frame_size = video_size + depth_size + intrinsic_size

    # Check if the entire frame data is available
    if len(buffer) < marker_index + VIDEO_HEADER_SIZE + frame_size:
        return None, buffer  # Incomplete frame data
    
    # Extract video data
    video_data = buffer[marker_index + VIDEO_HEADER_SIZE:marker_index + VIDEO_HEADER_SIZE + video_size]

    # Extract depth data
    depth_data = buffer[marker_index + VIDEO_HEADER_SIZE + video_size:marker_index + VIDEO_HEADER_SIZE + video_size + depth_size]

    # Extract intrinsic data
    intrinsic_buffer = buffer[marker_index + VIDEO_HEADER_SIZE + video_size + depth_size:marker_index + VIDEO_HEADER_SIZE + video_size + depth_size + intrinsic_size]
    fx, fy, cx, cy = struct.unpack("ffff", intrinsic_buffer) 

    # # Extract IMU data
    # imu_buffer = buffer[marker_index + HEADER_SIZE + video_size + depth_size + intrinsic_size:marker_index + HEADER_SIZE + frame_size]
    # r_ax, r_ay, r_az, r_gx, r_gy, r_gz, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = struct.unpack("dddddddddddd", imu_buffer)
    
    remaining_buffer = buffer[marker_index + VIDEO_HEADER_SIZE + frame_size:]
    return (timestamp, video_data, depth_data, fx, fy, cx, cy), remaining_buffer
            # r_ax, r_ay, r_az, r_gx, r_gy, r_gz,
            # acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z), remaining_buffer

def buffer_to_depth(frame_data):
    """
    Convert the raw depth buffer to a numpy array of float16 values
    with shape (DEPTH_HEIGHT, DEPTH_WIDTH) without normalization.
    """
    depth_map = np.frombuffer(frame_data, dtype=np.float16).copy()
    depth_map = depth_map.reshape((DEPTH_HEIGHT, DEPTH_WIDTH))  # Adjust shape as necessary


    # get rid of NaN and Inf values
    depth_map[np.isnan(depth_map)] = 0
    depth_map[np.isinf(depth_map)] = 0

    depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
    depth_8bit = depth_normalized.astype(np.uint8)
    return depth_8bit, depth_map

def display_loop():
    while True:
        video_frame = None
        depth_frame = None

        if not video_frame_queue.empty():
            video_frame = video_frame_queue.get()

        if not depth_frame_queue.empty():
            depth_frame = depth_frame_queue.get()

        if video_frame is not None:
            cv2.imshow("Video Frame", video_frame)

        if depth_frame is not None:
            cv2.imshow("Depth Map", depth_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def receive_imu_data(host, port):
    global recorded_count_imu, file_written_imu
    # create tcp socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"Starting server on {host}:{port}, waiting for a connection...")
    connection, client_address = server_socket.accept()
    print(f"Connection from {client_address}")
    try:
        while True:
            data_in = connection.recv(IMU_PACKET_SIZE)
            if not data_in:
                break
            if len(data_in) == IMU_PACKET_SIZE:
                # Extract data
                timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z  = struct.unpack("ddddddd", data_in[4:])

                if recorded_count_imu < DEBUG_FRAME_COUNTER_IMU:
                    recorded_frames_imu[recorded_count_imu]['timestamp'] = timestamp
                    recorded_frames_imu[recorded_count_imu]['acc'] = [acc_x, acc_y, acc_z]
                    recorded_frames_imu[recorded_count_imu]['gyro'] = [gyro_x, gyro_y, gyro_z]
                    print(f"IMU frame {recorded_count_imu+1}/{DEBUG_FRAME_COUNTER_IMU}")
                    recorded_count_imu += 1

                    # When 10 frames are recorded, write them to file in a separate thread
                    if recorded_count_imu == DEBUG_FRAME_COUNTER_IMU and not file_written_imu:
                        file_written_imu = True
                        threading.Thread(target=write_imu_frames_to_file).start()   

                print("IMU Data: @", timestamp)
                print("Acc Data:")
                print(f"{'Acc_X':<10}{'Acc_Y':<10}{'Acc_Z':<10}")  
                print(f"{acc_x:<10.6f}{acc_y:<10.6f}{acc_z:<10.6f}") 
                print("\n")
                print("Gyro Data:")
                print(f"{'Gyro_X':<10}{'Gyro_Y':<10}{'Gyro_Z':<10}")
                print(f"{gyro_x:<10.6f}{gyro_y:<10.6f}{gyro_z:<10.6f}")
                print("\n")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()
        print("Connection closed")


def receive_video(host, port):
    global recorded_count_video, file_written_video
    # Create and bind a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"Starting server on {host}:{port}, waiting for a connection...")
    connection, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    # Initialize H264 decoder
    decoder = h264decoder.H264Decoder()

    # Buffer to accumulate received data
    buffer = b''

    try:
        while True:
            # Receive data in chunks
            data_in = connection.recv(4096)
            if not data_in:
                break
            buffer += data_in

            while True:
                extracted_packet, buffer = extract_packet(buffer)
                if extracted_packet is None:
                    break  # Wait for more data

                (timestamp, video_data, depth_data, fx, fy, cx, cy) = extracted_packet


                # Process depth image: get the raw float16 depth values
                depth_img, depth_raw = buffer_to_depth(depth_data)
                if not depth_frame_queue.full():
                    depth_frame_queue.put(depth_img)

                # Decode H264 frame and process each decoded frame
                # YCbCr to RGB conversion
                framedatas = decoder.decode(video_data)
                for framedata in framedatas:
                    (frame, w, h, ls) = framedata
                    if frame is not None:
                        # Convert the frame to a numpy array and resize to 320x180
                        frame_array_video = np.frombuffer(frame, dtype=np.ubyte).reshape((h, ls // 3, 3))
                        frame_resized = cv2.resize(frame_array_video, (DEPTH_WIDTH, DEPTH_HEIGHT))

                        # rgb to bgr, cv2 uses BGR by default
                        bgr_frame = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

                        if not video_frame_queue.full():
                            video_frame_queue.put(bgr_frame)

                        # Record the frame if DEBUG_FRAME_COUNTER is not reached
                        if recorded_count_video < DEBUG_FRAME_COUNTER_VIDEO:
                            recorded_frames_video[recorded_count_video]['timestamp'] = timestamp
                            recorded_frames_video[recorded_count_video]['fx'] = fx
                            recorded_frames_video[recorded_count_video]['fy'] = fy
                            recorded_frames_video[recorded_count_video]['cx'] = cx
                            recorded_frames_video[recorded_count_video]['cy'] = cy
                            recorded_frames_video[recorded_count_video]['bgr'] = bgr_frame
                            recorded_frames_video[recorded_count_video]['depth'] = depth_raw
                            print(f"VIDEO frame {recorded_count_video+1}/{DEBUG_FRAME_COUNTER_VIDEO}")
                            recorded_count_video += 1

                            # When 10 frames are recorded, write them to file in a separate thread
                            if recorded_count_video == DEBUG_FRAME_COUNTER_VIDEO and not file_written_video:
                                file_written_video = True
                                threading.Thread(target=write_video_frames_to_file).start()
                    else:
                        print('frame is None')
                        break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()
        cv2.destroyAllWindows()
        print("Connection closed")

if __name__ == "__main__":
    # Get IP address
    ip_address = get_local_ip()
    print(f"Server IP address: {ip_address}")
    
    # Create and start the receive_data thread
    receive_video_thread = threading.Thread(target=receive_video, args=(HOST, VIDEO_PORT))
    receive_video_thread.start()

    receive_imu_thread = threading.Thread(target=receive_imu_data, args=(HOST, IMU_PORT))
    receive_imu_thread.start()

    # Main thread handles the GUI
    display_loop()

    # Wait for the data receiving thread to finish
    receive_imu_thread.join()
    receive_video_thread.join()
    print("Server shutting down.")
