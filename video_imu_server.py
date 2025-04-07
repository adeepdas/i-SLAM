import socket
import struct
import numpy as np
import cv2
import h264decoder  # Import the decoder from the cloned repository
import threading
import queue

START_MARKER = b'\xAB\xCD\xEF\x01'  # 4-byte unique identifier
HEADER_SIZE = 28  # 4 (marker) + 8 (timestamp) + 4 (video size) + 4 (depth size) + 4 (intrinsic size) + 4 (imu size)

# Server Config
ALL_DATA_PORT = 25005

DEPTH_HEIGHT = 180
DEPTH_WIDTH = 320

HOST = '0.0.0.0'  # Listen on all interfaces

DEBUG_FRAME_COUNTER = 60

video_frame_queue = queue.Queue(maxsize=10)
depth_frame_queue = queue.Queue(maxsize=10)

# --- Global variables for recording frames ---
# Define a structured dtype for each frame
frame_dtype = np.dtype([
    ('timestamp', np.float64),
    ('fx', np.float32),
    ('fy', np.float32),
    ('cx', np.float32),
    ('cy', np.float32),
    ('ax', np.float64),
    ('ay', np.float64),
    ('az', np.float64),
    ('gx', np.float64),
    ('gy', np.float64),
    ('gz', np.float64),
    ('rgb', np.uint8, (180, 320, 3)),   # 320x180 RGB image
    ('depth', np.float16, (180, 320))      # 320x180 depth image in raw float16
])

# Pre-allocate an array for 10 frames
recorded_frames = np.empty(DEBUG_FRAME_COUNTER, dtype=frame_dtype)
recorded_count = 0
file_written = False  # To ensure file write happens only once

def write_frames_to_file():
    # This function writes the recorded_frames array to a file.
    np.save('recorded_frames.npy', recorded_frames)
    print("Recorded frames saved to recorded_frames.npy")

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

    if len(buffer) < marker_index + HEADER_SIZE:
        return None, buffer  # Not enough data for a complete header

    # Extract header information
    header = buffer[marker_index:marker_index + HEADER_SIZE]
    timestamp, video_size, depth_size, intrinsic_size, imu_size = struct.unpack("dIIII", header[4:HEADER_SIZE])
    frame_size = video_size + depth_size + intrinsic_size + imu_size

    # Check if the entire frame data is available
    if len(buffer) < marker_index + HEADER_SIZE + frame_size:
        return None, buffer  # Incomplete frame data
    
    # Extract video data
    video_data = buffer[marker_index + HEADER_SIZE:marker_index + HEADER_SIZE + video_size]

    # Extract depth data
    depth_data = buffer[marker_index + HEADER_SIZE + video_size:marker_index + HEADER_SIZE + video_size + depth_size]

    # Extract intrinsic data
    intrinsic_buffer = buffer[marker_index + HEADER_SIZE + video_size + depth_size:marker_index + HEADER_SIZE + video_size + depth_size + intrinsic_size]
    fx, fy, cx, cy = struct.unpack("ffff", intrinsic_buffer) 

    # Extract IMU data
    imu_buffer = buffer[marker_index + HEADER_SIZE + video_size + depth_size + intrinsic_size:marker_index + HEADER_SIZE + frame_size]
    r_ax, r_ay, r_az, r_gx, r_gy, r_gz, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = struct.unpack("dddddddddddd", imu_buffer)
    
    remaining_buffer = buffer[marker_index + HEADER_SIZE + frame_size:]
    return (timestamp, video_data, depth_data, fx, fy, cx, cy,
            r_ax, r_ay, r_az, r_gx, r_gy, r_gz,
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z), remaining_buffer

def buffer_to_depth(frame_data):
    """
    Convert the raw depth buffer to a numpy array of float16 values
    with shape (DEPTH_HEIGHT, DEPTH_WIDTH) without normalization.
    """
    depth_map = np.frombuffer(frame_data, dtype=np.float16).copy()
    depth_map = depth_map.reshape((DEPTH_HEIGHT, DEPTH_WIDTH))  # Adjust shape as necessary
    # print(f"Depth map shape: {depth_map.shape}")
    # print("Min depth:", np.min(depth_map))
    # print("Max depth:", np.max(depth_map))
    # print("NaNs count:", np.isnan(depth_map).sum())
    # print("Infs count:", np.isinf(depth_map).sum())

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
            # For display purposes, we convert the raw float16 depth to a uint8 image
            # by normalizing it. This does not affect the recorded raw data.
            # depth_disp = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
            # depth_disp = depth_disp.astype(np.uint8)
            cv2.imshow("Depth Map", depth_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def receive_data(host, port):
    global recorded_count, file_written
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

                (timestamp, video_data, depth_data, fx, fy, cx, cy,
                 r_ax, r_ay, r_az, r_gx, r_gy, r_gz,
                 acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) = extracted_packet

                # print(f"Intrinsic: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                # print(f"IMU Data: @ {timestamp}")
                # print("Acc Data:")
                # print(f"{'Acc_X':<10}{'Acc_Y':<10}{'Acc_Z':<10}")
                # print(f"{acc_x:<10.6f}{acc_y:<10.6f}{acc_z:<10.6f}\n")
                # print("Gyro Data:")
                # print(f"{'Gyro_X':<10}{'Gyro_Y':<10}{'Gyro_Z':<10}")
                # print(f"{gyro_x:<10.6f}{gyro_y:<10.6f}{gyro_z:<10.6f}\n")

                # Process depth image: get the raw float16 depth values
                depth_img, depth_raw = buffer_to_depth(depth_data)
                if not depth_frame_queue.full():
                    depth_frame_queue.put(depth_img)

                # Decode H264 frame and process each decoded frame
                framedatas = decoder.decode(video_data)
                for framedata in framedatas:
                    (frame, w, h, ls) = framedata
                    if frame is not None:
                        # Convert the frame to a numpy array and resize to 320x180
                        frame_array_video = np.frombuffer(frame, dtype=np.ubyte).reshape((h, ls // 3, 3))
                        frame_resized = cv2.resize(frame_array_video, (320, 180))
                        if not video_frame_queue.full():
                            video_frame_queue.put(frame_resized)

                        # Record the frame if we haven't collected 10 frames yet
                        if recorded_count < DEBUG_FRAME_COUNTER:
                            recorded_frames[recorded_count]['timestamp'] = timestamp
                            recorded_frames[recorded_count]['fx'] = fx
                            recorded_frames[recorded_count]['fy'] = fy
                            recorded_frames[recorded_count]['cx'] = cx
                            recorded_frames[recorded_count]['cy'] = cy
                            # Use the acceleration and gyro data (acc_x, acc_y, acc_z and gyro_x, gyro_y, gyro_z)
                            recorded_frames[recorded_count]['ax'] = acc_x
                            recorded_frames[recorded_count]['ay'] = acc_y
                            recorded_frames[recorded_count]['az'] = acc_z
                            recorded_frames[recorded_count]['gx'] = gyro_x
                            recorded_frames[recorded_count]['gy'] = gyro_y
                            recorded_frames[recorded_count]['gz'] = gyro_z
                            recorded_frames[recorded_count]['rgb'] = frame_resized
                            recorded_frames[recorded_count]['depth'] = depth_raw
                            print(f"Recorded frame {recorded_count+1}/10")
                            recorded_count += 1

                            # When 10 frames are recorded, write them to file in a separate thread
                            if recorded_count == DEBUG_FRAME_COUNTER and not file_written:
                                file_written = True
                                threading.Thread(target=write_frames_to_file).start()
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
    receive_data_thread = threading.Thread(target=receive_data, args=(HOST, ALL_DATA_PORT))
    receive_data_thread.start()

    # Main thread handles the GUI
    display_loop()

    # Wait for the data receiving thread to finish
    receive_data_thread.join()
    print("Server shutting down.")
