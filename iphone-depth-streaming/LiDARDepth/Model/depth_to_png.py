import numpy as np
import cv2

# Input and output file names
input_file = "depth_log.txt"
output_txt_file = "depth_array.txt"
output_image_file = "depth_map_color.png"

def parse_depth_file(filename):
    """Reads a depth log file and extracts depth values into a dictionary."""
    depth_data = {}
    with open(filename, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                x, y, depth = int(parts[0]), int(parts[1]), float(parts[2])
                depth_data[(x, y)] = depth
    return depth_data

def create_depth_map(depth_data):
    """Converts the depth dictionary into a 2D NumPy array."""
    if not depth_data:
        raise ValueError("No depth data found!")

    max_x = max(k[0] for k in depth_data.keys()) + 1
    max_y = max(k[1] for k in depth_data.keys()) + 1
    depth_map = np.zeros((max_x, max_y))

    for (x, y), depth in depth_data.items():
        depth_map[x, y] = depth

    return depth_map

def save_color_depth_map(depth_map, txt_filename, img_filename):
    """Saves the depth map as a text file and a color-coded image."""
    np.savetxt(txt_filename, depth_map, fmt="%.3f")

    # Normalize depth values to 0-255 for color mapping
    depth_min, depth_max = np.min(depth_map), np.max(depth_map)
    depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    # Apply a color map: Red (near) -> Blue (far)
    depth_colored = cv2.applyColorMap(255 - depth_normalized, cv2.COLORMAP_JET)

    cv2.imwrite(img_filename, depth_colored)

def main():
    depth_data = parse_depth_file(input_file)
    depth_map = create_depth_map(depth_data)
    save_color_depth_map(depth_map, output_txt_file, output_image_file)
    print(f"Saved depth array to {output_txt_file} and color-coded depth map to {output_image_file}")

if __name__ == "__main__":
    main()
