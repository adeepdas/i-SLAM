import socket
import struct
import numpy as np
import cv2
import h264decoder  # Import the decoder from the cloned repository
import threading

START_MARKER = b'\xAB\xCD\xEF\x01'  # 4-byte unique identifier
HEADER_SIZE = 16  # 4 (marker) + 8 (timestamp) + 4 (frame size)


# Server Config
VIDEO_PORT = 12005
IMU_PORT = 13005
HOST = '0.0.0.0'  # Listen on all interfaces


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
                # print(f"Received frame at {timestamp:.6f} seconds, size: {len(frame_data)} bytes")

                # Decode H264 frame
                framedatas = decoder.decode(frame_data)
                
                for framedata in framedatas:
                    (frame, w, h, ls) = framedata
                    if frame is not None:
                        print("Frame timestamp is: ", timestamp)
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
            data_in = connection.recv(116)
            if not data_in:
                break
            if len(data_in) == 116:
                # Extract data
                timestamp, ax, ay, az, gx, gy, gz, mx, my, mz, qx, qy, qz, qw = struct.unpack("dddddddddddddd", data_in[4:])
                # Print the data in a nice table format
                print(f"{'Timestamp':<15}{'Ax':<10}{'Ay':<10}{'Az':<10}{'Gx':<10}{'Gy':<10}{'Gz':<10}{'Mx':<10}{'My':<10}{'Mz':<10}{'Qx':<10}{'Qy':<10}{'Qz':<10}{'Qw':<10}")
                print(f"{timestamp:<15.6f}{ax:<10.6f}{ay:<10.6f}{az:<10.6f}{gx:<10.6f}{gy:<10.6f}{gz:<10.6f}{mx:<10.3f}{my:<10.3f}{mz:<10.3f}{qx:<10.6f}{qy:<10.6f}{qz:<10.6f}{qw:<10.6f}")
                

    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()
        print("Connection closed")



if __name__ == "__main__":
    # Create threads for video & IMU reception
    video_thread = threading.Thread(target=receive_h264_video, args=(HOST, VIDEO_PORT))
    imu_thread = threading.Thread(target=receive_imu_data, args=(HOST, IMU_PORT))

    # Start both threads
    video_thread.start()
    imu_thread.start()

    # Wait for both to finish
    video_thread.join()
    imu_thread.join()

    print("Server shutting down.")


