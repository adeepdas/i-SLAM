import socket
import struct
import numpy as np
import cv2
import h264decoder  # Import the decoder from the cloned repository
import threading

START_MARKER = b'\xAB\xCD\xEF\x01'  # 4-byte unique identifier
HEADER_SIZE = 16  # 4 (marker) + 8 (timestamp) + 4 (frame size)

IMU_PACKET_SIZE = 212 # 4 (marker) + 8 (timestamp) + 200 (data)

DEPTH_FRAME_SIZE = 768 * 576 * 2  # 768x576 image with float16 depth values

# Server Config
VIDEO_PORT = 12005
IMU_PORT = 13005
HOST = '0.0.0.0'  # Listen on all interfaces
DEPTH_PORT = 14005

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
    # print(f"Header: {header}")
    timestamp, frame_size = struct.unpack("dI", header[4:16])

    # print(f"Timestamp: {timestamp}, Frame size: {frame_size}")

    # Check if the entire frame data is available
    # print(f"Buffer size in extract: {len(buffer)} bytes")
    if len(buffer) < marker_index + HEADER_SIZE + frame_size:
        # print(f"Error in Extracted Packer: Buffer size in depth: {len(buffer)} bytes")
        return None, buffer  # Incomplete frame data
    

    # Extract the frame data
    # Extract the frame data
    frame_data = buffer[marker_index + HEADER_SIZE:marker_index + HEADER_SIZE + frame_size]
    remaining_buffer = buffer[marker_index + HEADER_SIZE + frame_size:]

    return (timestamp, frame_data), remaining_buffer


def receive_h264_video(host, port):
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

                timestamp, frame_data = extracted_packet
                # print(f"Received frame at {timestamp:.6f} seconds, size: {len(frame_data)} bytes")

                # Decode H264 frame
                framedatas = decoder.decode(frame_data)
                
                for framedata in framedatas:
                    (frame, w, h, ls) = framedata
                    if frame is not None:
                        # print("Frame timestamp is: ", timestamp)
                        # Convert the frame to a numpy array
                        frame = np.frombuffer(frame, dtype=np.ubyte).reshape((h, ls // 3, 3))
                        # Display the decoded frame
                        cv2.imshow('Frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        print('frame is None')
                        break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()
        cv2.destroyAllWindows()
        print("Connection closed")

def receive_imu_data(host, port):
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
            # print(f"Received {len(data_in)} bytes of IMU data")
            if len(data_in) == IMU_PACKET_SIZE:
                # Extract data
                timestamp, r_ax, r_ay, r_az, r_gx, r_gy, r_gz, r_mx, r_my, r_mz, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, yaw, pitch, roll, quat_x, quat_y, quat_z, quat_w = struct.unpack("dddddddddddddddddddddddddd", data_in[4:])
                
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

def buffer_to_image(frame_data):

    depth_map = np.frombuffer(frame_data, dtype=np.float16)
    # print(f"Depth map shape before reshape: {depth_map.shape}")
    depth_map = depth_map.reshape((240, 320))  # Adjust shape as necessary

    # print(f"Depth map shape: {depth_map.shape}")

    # print("Min depth:", np.min(depth_map))
    # print("Max depth:", np.max(depth_map))
    # print("NaNs count:", np.isnan(depth_map).sum())
    # print("Infs count:", np.isinf(depth_map).sum())


    depth_map = depth_map.astype(np.float32)  # Convert float16 → float32

            # Normalize from (0.3472, 1.698) to (0, 255)
    depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
    depth_8bit = depth_normalized.astype(np.uint8)

    # cv2.imshow("Depth 8bit", depth_8bit)
    # cv2.imwrite("depth_8bit.png", depth_8bit)
    # depth_colored = cv2.applyColorMap(255-depth_8bit, cv2.COLORMAP_JET)
    # # cv2.imshow("Depth Colored", depth_colored)
    # cv2.imwrite("depth_colored.png", depth_colored)
    # Save as PNG
    # cv2.imwrite("depth_map.png", depth_8bit)

    # Optional: Show it in a window
    cv2.imshow("Depth Map", depth_8bit)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()


def receive_depth_data(host, port):
 # Create and bind a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    
    print(f"Starting server on {host}:{port}, waiting for a connection...")
    connection, client_address = server_socket.accept()
    print(f"Connection from {client_address}")

    
    # Buffer to accumulate received data
    buffer = b''
    counter = 0

    try:
        while True:
            # Receive data in chunks
            data_in = connection.recv(4096)

            # print(f"Received {len(data_in)} bytes of depth data")
            if not data_in:
                # print("No data received, breaking the loop.")
                break
            buffer += data_in

            while True:
                extracted_packet, buffer = extract_packet(buffer)
                if extracted_packet is None:
                    # print(f"Error: Buffer size in depth: {len(buffer)} bytes")
                    break  # Wait for more data

                timestamp, frame_data = extracted_packet

                # print(f"Received depth frame at {timestamp:.6f} seconds, size: {len(frame_data)} bytes")
                buffer_to_image(frame_data)




    




    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()
        print("Connection closed")



if __name__ == "__main__":
    # get ip address
    ip_address = get_local_ip()
    print(f"Server IP address: {ip_address}")
    
    # Create threads for video & IMU reception
    video_thread = threading.Thread(target=receive_h264_video, args=(HOST, VIDEO_PORT))
    imu_thread = threading.Thread(target=receive_imu_data, args=(HOST, IMU_PORT))
    depth_thread = threading.Thread(target=receive_depth_data, args=(HOST, DEPTH_PORT))

    # Start both threads
    video_thread.start()
    imu_thread.start()
    depth_thread.start()

    # Wait for both to finish
    video_thread.join()
    imu_thread.join()
    depth_thread.join()

    print("Server shutting down.")


