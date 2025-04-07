import socket
import struct
import numpy as np
import cv2
import h264decoder  # Import the decoder from the cloned repository
import threading
import queue
import time
import csv


START_MARKER = b'\xAB\xCD\xEF\x01'  # 4-byte unique identifier
HEADER_SIZE = 16  # 4 (marker) + 8 (timestamp) + 4 (frame size)
IMU_PACKET_SIZE = 212 # 4 (marker) + 8 (timestamp) + 200 (data)

# Server Config
VIDEO_PORT = 12005
IMU_PORT = 13005
DEPTH_PORT = 14005

DEPTH_HEIGHT = 180
DEPTH_WIDTH = 320

TIME_THRESHOLD = 10  # 0.05 milliseconds is lowest without file writing
MAX_FRAMES_TO_SAVE = 10  # save the first 10 frames

frame_save_counter = 0  # Global counter for saved frames

HOST = '0.0.0.0'  # Listen on all interfaces

video_frame_queue = queue.Queue(maxsize=10)
depth_frame_queue = queue.Queue(maxsize=10)
writer_queue = queue.Queue()

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
    timestamp, frame_size = struct.unpack("dI", header[4:16])


    # Check if the entire frame data is available
    if len(buffer) < marker_index + HEADER_SIZE + frame_size:
        return None, buffer  # Incomplete frame data
    
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

                # Decode H264 frame
                framedatas = decoder.decode(frame_data)
                
                for framedata in framedatas:
                    (frame, w, h, ls) = framedata
                    if frame is not None:
                        # Convert the frame to a numpy array
                        frame = np.frombuffer(frame, dtype=np.ubyte).reshape((h, ls // 3, 3))
                        # Convert frame into float 16 values
                        # frame = frame.astype(np.float16)

                        video_frame_resize = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                        if not video_frame_queue.full():
                            video_frame_queue.put((timestamp, video_frame_resize))                        

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
    depth_map = np.frombuffer(frame_data, dtype=np.float16).copy()
    depth_map = depth_map.reshape((DEPTH_HEIGHT, DEPTH_WIDTH))  # Adjust shape as necessary
    print(f"Depth map shape: {depth_map.shape}")
    # print("Min depth:", np.min(depth_map))
    # print("Max depth:", np.max(depth_map))
    # print("NaNs count:", np.isnan(depth_map).sum())
    # print("Infs count:", np.isinf(depth_map).sum())

    # convert all nan and inf values to 0
    depth_map[np.isnan(depth_map)] = 0
    depth_map[np.isinf(depth_map)] = 0
    print("NaNs count after:", np.isnan(depth_map).sum())

    depth_map = depth_map.astype(np.float32)  # Convert float16 → float32
    depth_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map)) * 255
    depth_8bit = depth_normalized.astype(np.uint8)
    return depth_8bit

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

            if not data_in:
                break
            buffer += data_in

            while True:
                extracted_packet, buffer = extract_packet(buffer)
                if extracted_packet is None:
                    break  # Wait for more data
                timestamp, frame_data = extracted_packet
                # print(f"Received Depth frame size: {len(frame_data)}")
                # buffer_to_image(frame_data)
                depth_img = buffer_to_image(frame_data)
                if not depth_frame_queue.full():
                    depth_frame_queue.put((timestamp, depth_img))

    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()
        print("Connection closed")

def get_synchronized_frame():
    """
    Retrieves a synchronized pair of video and depth frames from their respective queues.
    Returns:
        video_frame (np.array): The video frame (shape: 180,320,3)
        depth_frame (np.array): The depth frame (shape: 180,320)
        timestamp (float): The video frame's timestamp used for synchronization.
    The function waits until frames from both queues are available and their timestamps differ
    by no more than TIME_THRESHOLD.
    """
    while True:
        try:
            video_timestamp, video_frame = video_frame_queue.get_nowait()
            depth_timestamp, depth_frame = depth_frame_queue.get_nowait()
            
            # Check if timestamps are close enough
            if abs(video_timestamp - depth_timestamp) <= TIME_THRESHOLD:
                return video_timestamp, video_frame, depth_frame 
            else:
                # Discard the older frame and continue searching
                if video_timestamp < depth_timestamp:
                    continue  # discard this video frame
                else:
                    continue  # discard this depth frame
        except queue.Empty:
            time.sleep(0.01)
            continue


def write_npz(timestamps, combined_frames):
    """
    Write the accumulated timestamps and combined frames to one NPZ file.
    The NPZ file will contain:
      - 'timestamps': an array of float64 timestamps
      - 'frames': an array of shape (N, 180, 320, 4) in float16
    """
    np.savez("combined_frames.npz",
             timestamps=np.array(timestamps, dtype=np.float64),
             frames=np.array(combined_frames, dtype=np.float16))
    print("Wrote 10 synchronized frames to 'combined_frames.npz'")

def write_csv(timestamps, combined_frames):
    """
    Write the accumulated timestamps and combined frames to a CSV file.
    Each row will contain:
      - The timestamp (float64) in the first column
      - The flattened combined frame data (180*320*4 = 230400 values as float16)
    Note: The CSV file will be very wide.
    """
    csv_filename = "combined_frames.csv"
    num_pixels = 180 * 320 * 4  # Total pixel values per combined frame
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header row (optional)
        header = ["timestamp"] + [f"pixel_{i}" for i in range(num_pixels)]
        writer.writerow(header)
        
        for ts, frame in zip(timestamps, combined_frames):
            flattened = frame.flatten()
            row = [ts] + flattened.tolist()
            writer.writerow(row)
    print(f"Wrote 10 synchronized frames to '{csv_filename}'")

def display_and_save_loop():
    """
    Continuously retrieves synchronized frames, displays the video and depth frames in separate windows,
    and accumulates the first 10 combined frames (converted to float16). Once 10 frames have been accumulated,
    their data is pushed to the writer_queue for file writing. The display loop then continues to show new frames.
    """
    timestamps = []
    combined_frames = []
    file_written = False  # Flag to ensure we only queue file writing once

    while True:
        # Retrieve a synchronized frame

        timestamp, video_frame, depth_frame = get_synchronized_frame()
        cv2.imshow("Video Frame", video_frame)
        cv2.imshow("Depth Frame", depth_frame) 
        # Display the separate frames

        
        # # Convert frames to float16
        # video_fp16 = video_frame.astype(np.float16)
        # depth_fp16 = depth_frame.astype(np.float16)
        # # # Expand depth frame to shape (180,320,1) for concatenation
        # depth_fp16 = np.expand_dims(depth_fp16, axis=2)
        # # Concatenate to form a combined frame: shape (180,320,4)
        # combined = np.concatenate((video_fp16, depth_fp16), axis=2)
        
        # # Accumulate the frame and timestamp if under the limit
        # if len(timestamps) < MAX_FRAMES_TO_SAVE:
        #     timestamps.append(timestamp)
        #     combined_frames.append(combined)
        #     print(f"Accumulated frame {len(timestamps)}")
        
        # # Once 10 frames have been collected and file writing hasn't been triggered yet,
        # # push the data to the writer queue.
        # if len(timestamps) == MAX_FRAMES_TO_SAVE and not file_written:
        #     print("Writing 10 frames to file...")
        #     writer_queue.put((timestamps, combined_frames))
        #     file_written = True
        
        # Continue displaying frames until 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

def file_writer():
    """
    Worker thread that waits for data on the writer_queue.
    When it gets data (timestamps and combined frames), it writes the NPZ and CSV files.
    """
    while True:
        data = writer_queue.get()
        if data is None:  # Sentinel to stop the thread.
            break
        timestamps, combined_frames = data
        write_npz(timestamps, combined_frames)
        # write_csv(timestamps, combined_frames)
        writer_queue.task_done()

if __name__ == "__main__":
    # get ip address
    ip_address = get_local_ip()
    print(f"Server IP address: {ip_address}")
    
    # Create threads for video & IMU reception
    video_thread = threading.Thread(target=receive_h264_video, args=(HOST, VIDEO_PORT))
    imu_thread = threading.Thread(target=receive_imu_data, args=(HOST, IMU_PORT))
    depth_thread = threading.Thread(target=receive_depth_data, args=(HOST, DEPTH_PORT))
    # Create and start the writer thread.
    writer_thread = threading.Thread(target=file_writer, daemon=True)
    writer_thread.start()

    # Start both threads
    video_thread.start()
    # imu_thread.start()
    depth_thread.start()
    writer_queue.put(None)
    
    # Main thread handles GUI
    display_and_save_loop()

    # Wait for both to finish
    video_thread.join()
    # imu_thread.join()
    depth_thread.join()
    
    writer_thread.join()
    print("Server shutting down.")


