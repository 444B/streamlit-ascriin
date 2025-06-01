import cv2
import numpy as np
import os
import time
import random

# --- Configuration ---
ASCII_WIDTH = 100  # Width of the output in characters
CAMERA_INDEX = 2   # 0 for default camera, change if you have multiple

# --- Outline Detection Settings ---
GAUSSIAN_BLUR_KERNEL_SIZE = (5, 5) # Kernel size for Gaussian blur (must be odd numbers)
CANNY_THRESHOLD1 = 50              # Lower threshold for Canny edge detection
CANNY_THRESHOLD2 = 150             # Higher threshold for Canny edge detection
OUTLINE_CHAR = '#'                 # Character to use for detected outlines

# --- Matrix Rain Settings ---
# MATRIX_FONT_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
# For a more classic Matrix feel, you could try Katakana (ensure your terminal supports it):
MATRIX_FONT_CHARS = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヰヱヲン"
MATRIX_FALL_SPEED_FRAMES = 1 # Update matrix rain every N camera frames. Lower is faster rain.
MATRIX_SPAWN_PROBABILITY = 0.15 # Chance for a new drop to start in an empty column top

# --- ANSI Color Codes ---
GREEN = "\033[32m"
BRIGHT_WHITE = "\033[97m"
RESET_COLOR = "\033[0m"

# --- Global State for Matrix ---
matrix_columns = [] # Will be initialized in main
# Each element: {'y': int, 'char': str, 'active': bool, 'current_fall_speed_frames': int}
output_height = 0 # Will be calculated based on aspect ratio

# --- Helper Functions ---

def clear_console():
    """Clears the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def initialize_matrix_rain(width, height):
    """Initializes the state for the matrix rain effect."""
    global matrix_columns, output_height
    output_height = height # Store calculated height
    matrix_columns = []
    for _ in range(width):
        matrix_columns.append({
            'y': 0,
            'char': random.choice(MATRIX_FONT_CHARS),
            'active': False,
            'current_fall_speed_frames': 0 # Counter for fall speed
        })

def update_matrix_rain_state():
    """Updates the positions and characters of the matrix rain."""
    global matrix_columns, output_height
    if not matrix_columns or output_height == 0: # Ensure initialized
        return

    for i in range(len(matrix_columns)):
        col = matrix_columns[i]
        if col['active']:
            col['current_fall_speed_frames'] += 1
            if col['current_fall_speed_frames'] >= MATRIX_FALL_SPEED_FRAMES:
                col['y'] += 1
                col['current_fall_speed_frames'] = 0
                # Change character as it falls for more dynamism (optional)
                if random.random() < 0.1: # Small chance to change char mid-fall
                     col['char'] = random.choice(MATRIX_FONT_CHARS)

                if col['y'] >= output_height:
                    col['active'] = False
                    col['y'] = 0 # Reset y for next spawn
        else:
            # Try to spawn a new drop
            if random.random() < MATRIX_SPAWN_PROBABILITY:
                col['active'] = True
                col['y'] = 0
                col['char'] = random.choice(MATRIX_FONT_CHARS)
                col['current_fall_speed_frames'] = 0


def get_frame_aspect_ratio_height(frame_width_pixels, frame_height_pixels, target_ascii_width):
    """Calculates ASCII height maintaining aspect ratio, adjusted for char shape."""
    aspect_ratio = frame_height_pixels / float(frame_width_pixels)
    # 0.55 is an empirical correction for typical character aspect ratio (taller than wide)
    # You might need to adjust this for your terminal font.
    ascii_height = int(aspect_ratio * target_ascii_width * 0.55)
    return ascii_height

def process_camera_frame_to_outline_data(frame, target_width, target_height):
    """
    Processes a camera frame to detect outlines.
    Returns a 2D list (target_height x target_width) where cells contain
    OUTLINE_CHAR for edges or None for background.
    """
    # 1. Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian Blur for noise reduction
    blurred_frame = cv2.GaussianBlur(gray_frame, GAUSSIAN_BLUR_KERNEL_SIZE, 0)

    # 3. Apply Canny edge detection
    edges = cv2.Canny(blurred_frame, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    # 4. Resize edge map to target ASCII dimensions
    # Ensure target_height is > 0 before resizing
    if target_height <= 0 or target_width <= 0:
        return [[None for _ in range(target_width)] for _ in range(max(1, target_height))] # Return empty if invalid dims

    resized_edges = cv2.resize(edges, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    # 5. Convert to outline data structure
    outline_data = [[None for _ in range(target_width)] for _ in range(target_height)]
    for r in range(target_height):
        for c in range(target_width):
            if resized_edges[r, c] > 0: # Edge pixels are non-zero in Canny output
                outline_data[r][c] = OUTLINE_CHAR
    return outline_data

# --- Main ---
def main():
    """Main function to capture video and display ASCII art with matrix effect."""
    global output_height # To use the globally set output_height

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {CAMERA_INDEX}.")
        return

    # Get first frame to determine aspect ratio and initialize matrix
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame from camera.")
        cap.release()
        return
    
    frame_h_pixels, frame_w_pixels, _ = frame.shape
    calculated_height = get_frame_aspect_ratio_height(frame_w_pixels, frame_h_pixels, ASCII_WIDTH)
    if calculated_height <= 0:
        print(f"Error: Calculated ASCII height is {calculated_height}. Please check ASCII_WIDTH or camera feed.")
        cap.release()
        return
        
    initialize_matrix_rain(ASCII_WIDTH, calculated_height) # Initialize with calculated height

    print(f"Starting ASCII camera with Matrix Effect. Settings:")
    print(f"  - ASCII Dimensions: {ASCII_WIDTH}w x {output_height}h")
    print(f"  - Outline Char: '{OUTLINE_CHAR}' (Bright White)")
    print(f"  - Matrix Chars: Green, Speed ~{MATRIX_FALL_SPEED_FRAMES} frame(s)/step")
    print("Press 'q' in the 'Input Feed' window to quit.")
    time.sleep(2)

    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Process camera frame for outlines
            outline_data = process_camera_frame_to_outline_data(frame, ASCII_WIDTH, output_height)

            # Update matrix rain state (can be tied to frame_count for different speed)
            # if frame_count % MATRIX_FALL_SPEED_FRAMES == 0: # This was moved into update_matrix_rain_state
            update_matrix_rain_state()

            # --- Render the combined frame ---
            clear_console()
            display_buffer = []
            
            for r in range(output_height):
                line_str_parts = []
                current_color_on_line = None
                for c in range(ASCII_WIDTH):
                    char_to_print = ' ' # Default to space
                    color_to_use = GREEN  # Default to matrix green

                    is_outline = outline_data[r][c] == OUTLINE_CHAR
                    
                    if is_outline:
                        char_to_print = OUTLINE_CHAR
                        color_to_use = BRIGHT_WHITE
                    elif matrix_columns[c]['active'] and matrix_columns[c]['y'] == r : # Check if matrix char is at this specific row
                        char_to_print = matrix_columns[c]['char']
                        color_to_use = GREEN # Already default, but explicit
                    # else: it remains a space in green (or could be empty if matrix density is low)

                    # Optimize ANSI code changes
                    if color_to_use != current_color_on_line:
                        if current_color_on_line is not None: # Only if a color was active
                             pass # No need to reset if immediately setting new color
                        line_str_parts.append(color_to_use)
                        current_color_on_line = color_to_use
                    
                    line_str_parts.append(char_to_print)

                if current_color_on_line is not None: # Ensure color is reset at end of line
                    line_str_parts.append(RESET_COLOR)
                display_buffer.append("".join(line_str_parts))
            
            for line in display_buffer:
                print(line)
            
            # Display a small window for key presses and original feed
            preview_frame = cv2.resize(frame, (200, int(200 * (frame_h_pixels/frame_w_pixels))))
            cv2.imshow('Input Feed (Press Q to quit)', preview_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1

    except KeyboardInterrupt:
        print("\nExiting due to Ctrl+C...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        clear_console()
        print("ASCII Camera with Matrix Effect stopped.")

if __name__ == "__main__":
    main()