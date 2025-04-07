import socket
import struct
import numpy as np
import cv2
import h264decoder  # Import the decoder from the cloned repository
import threading
import queue


START_MARKER = b'\xAB\xCD\xEF\x01'  # 4-byte unique identifier
HEADER_SIZE = 28  # 4 (marker) + 8 (timestamp) + 4 (video size) + 4 (depth size) + 4 (intrinsic Size) + 4 (imu size

# Server Config
ALL_DATA_PORT = 25005

DEPTH_HEIGHT = 180
DEPTH_WIDTH = 320

HOST = '0.0.0.0'  # Listen on all interfaces

video_frame_queue = queue.Queue(maxsize=10)
depth_frame_queue = queue.Queue(maxsize=10)

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
    

    # Extract the frame data
    # frame_data = buffer[marker_index + HEADER_SIZE:marker_index + HEADER_SIZE + frame_size]
    remaining_buffer = buffer[marker_index + HEADER_SIZE + frame_size:]

    return (timestamp, video_data, depth_data, fx, fy, cx, cy, r_ax, r_ay, r_az, r_gx, r_gy, r_gz, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z), remaining_buffer



def buffer_to_image(frame_data):
    depth_map = np.frombuffer(frame_data, dtype=np.float16)
    depth_map = depth_map.reshape((DEPTH_HEIGHT, DEPTH_WIDTH))  # Adjust shape as necessary
    # print(f"Depth map shape: {depth_map.shape}")
    # print("Min depth:", np.min(depth_map))
    # print("Max depth:", np.max(depth_map))
    # print("NaNs count:", np.isnan(depth_map).sum())
    # print("Infs count:", np.isinf(depth_map).sum())

    depth_map = depth_map.astype(np.float32)  # Convert float16 → float32
    depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
    depth_8bit = depth_normalized.astype(np.uint8)
    return depth_8bit

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


def receive_data(host, port):
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

                timestamp, video_data, depth_data, fx, fy, cx, cy, r_ax, r_ay, r_az, r_gx, r_gy, r_gz, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = extracted_packet


                # print fx, fy, cx, cy
                print(f"Intrinsic: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

                print(f"IMU Data: @ {timestamp}")
                print("Acc Data:")
                print(f"{'Acc_X':<10}{'Acc_Y':<10}{'Acc_Z':<10}")
                print(f"{acc_x:<10.6f}{acc_y:<10.6f}{acc_z:<10.6f}")
                print("\n")
                print("Gyro Data:")
                print(f"{'Gyro_X':<10}{'Gyro_Y':<10}{'Gyro_Z':<10}")
                print(f"{gyro_x:<10.6f}{gyro_y:<10.6f}{gyro_z:<10.6f}")
                print("\n")

                # Display depth data
                depth_img = buffer_to_image(depth_data)
                if not depth_frame_queue.full():
                    depth_frame_queue.put(depth_img)

                # Decode H264 frame and display
                framedatas = decoder.decode(video_data)
                for framedata in framedatas:
                    (frame, w, h, ls) = framedata
                    if frame is not None:
                        # Convert the frame to a numpy array
                        frame = np.frombuffer(frame, dtype=np.ubyte).reshape((h, ls // 3, 3))
                        # reseize the frame to 320x180
                        frame = cv2.resize(frame, (320, 180))
                        if not video_frame_queue.full():
                            video_frame_queue.put(frame)                        
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
    # get ip address
    ip_address = get_local_ip()
    print(f"Server IP address: {ip_address}")
    
    # Create threads for recieving data
    receive_data_thread = threading.Thread(target=receive_data, args=(HOST, ALL_DATA_PORT))

    # Start thread
    receive_data_thread.start()

    # Main thread handles GUI
    display_loop()

    # Wait for thread to finish
    receive_data_thread.join()
    print("Server shutting down.")


