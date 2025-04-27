"""
Improved Parking Space Detector
Uses background subtraction and contour analysis to detect free parking spaces
"""
import cv2
import numpy as np
import yaml
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab
import time
from plyer import notification

class MotionDetector:
    """Detects motion using background subtraction"""
    
    def __init__(self, history=500, threshold=20):
        # Initialize background subtractor
        # History: number of frames to build the background model
        # Threshold: how different a pixel must be to be considered foreground
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=threshold,
            detectShadows=True
        )
        
    def detect(self, frame):
        """Apply background subtraction to detect motion"""
        # Apply background subtractor
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (set to 0 instead of 127)
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        return fg_mask

class ParkingSpaceDetector:
    """Detects free parking spaces in images or video"""
    
    def __init__(self):
        # Settings
        self.notification_cooldown = 60  # Seconds between notifications
        self.last_notification_time = 0
        self.cap = None
        self.input_source = None
        self.threshold_value = 50  # Threshold for detection (higher = more sensitive)
        self.spaces = []
        
        # Motion detector for movement tracking (supplementary)
        self.motion_detector = MotionDetector(history=50, threshold=16)
        
        # Frame preprocessing
        self.use_blur = True
        self.blur_size = 5
        
        # Display scaling
        self.display_scale = 1.0
        
        # Reference frame for static analysis
        self.reference_frame = None
    
    def load_spaces(self, filename='parking_spaces.yml'):
        """Load saved parking spaces from YAML file"""
        try:
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
                
            # Load spaces
            self.spaces = []
            for space_data in data.get('spaces', []):
                points_lists = space_data.get('points', [])
                if points_lists:
                    # Convert each list to tuple for consistency
                    points_tuples = [tuple(point) for point in points_lists]
                    self.spaces.append(points_tuples)
                    
            # Get image dimensions
            dimensions = data.get('image_dimensions', {})
            self.original_width = dimensions.get('width', 0)
            self.original_height = dimensions.get('height', 0)
            self.display_scale = dimensions.get('scale', 1.0)
                
            print(f"Loaded {len(self.spaces)} parking spaces")
            return True
        except Exception as e:
            print(f"Error loading spaces: {e}")
            self.spaces = []
            return False
    
    def choose_input_source(self):
        """Show dialog to select input source (webcam or screen)"""
        # Create dialog window
        root = tk.Tk()
        root.title("Select Input")
        root.geometry("300x200")
        root.configure(bg="#f0f0f0")
        
        # Add title
        tk.Label(root, text="Choose Input Source", font=("Arial", 14), bg="#f0f0f0").pack(pady=15)
        
        # Button callbacks
        def select_webcam():
            self.input_source = 'webcam'
            root.destroy()
        
        def select_screen():
            self.input_source = 'screen'
            root.destroy()
        
        # Create buttons
        tk.Button(root, text="Webcam", command=select_webcam, 
                  bg="#4CAF50", fg="white", width=12, font=("Arial", 12)).pack(pady=10)
        tk.Button(root, text="Screen", command=select_screen, 
                  bg="#2196F3", fg="white", width=12, font=("Arial", 12)).pack(pady=10)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_width()) // 2
        y = (root.winfo_screenheight() - root.winfo_height()) // 2
        root.geometry(f"+{x}+{y}")
        
        # Show dialog and wait for selection
        root.mainloop()
        
        # If webcam selected, initialize it
        if self.input_source == 'webcam':
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam")
                return False
        
        return self.input_source is not None
    
    def preprocess_frame(self, frame):
        """Preprocess a frame for better detection"""
        # Keep the original color image for HSV analysis
        # Just apply blur to reduce noise if enabled
        if self.use_blur:
            frame = cv2.GaussianBlur(frame, (self.blur_size, self.blur_size), 0)
            
        return frame
    
    def detect_vehicle_static(self, frame, space_mask):
        """Detect vehicles using static image analysis (better suited for aerial view)"""
        # Make sure we're working with a color image
        if len(frame.shape) < 3:
            # Convert grayscale to BGR if needed
            frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame
            
        # Extract the masked region from the color image
        masked_area = cv2.bitwise_and(frame_color, frame_color, mask=space_mask)
        
        # 1. Color Analysis - parking lots are usually gray, cars have more color variance
        hsv = cv2.cvtColor(masked_area, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Calculate saturation variance (higher for vehicles, lower for empty spaces)
        s_mean, s_std = cv2.meanStdDev(s, mask=space_mask)
        s_variance = s_std[0][0]
        
        # 2. Shadow detection - vehicles cast shadows, empty spaces don't
        # High contrast in value channel indicates shadows
        v_mean, v_std = cv2.meanStdDev(v, mask=space_mask)
        v_contrast = v_std[0][0]
        
        # 3. Edge density - vehicles have more edges than empty pavement
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        masked_edges = cv2.bitwise_and(edges, edges, mask=space_mask)
        edge_pixels = cv2.countNonZero(masked_edges)
        total_pixels = cv2.countNonZero(space_mask)
        edge_percentage = (edge_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # 4. Mean intensity - vehicles tend to be darker than pavement 
        # (especially in aerial views)
        mean_intensity = cv2.mean(gray, mask=space_mask)[0]
        # Normalize and invert (darker = higher score)
        darkness_score = (255 - mean_intensity) / 255 * 100
        
        # Combine scores with appropriate weights
        # These weights are calibrated for aerial parking lot views
        combined_score = (
            darkness_score * 0.5 +      # Darkness (strongest indicator)
            s_variance * 0.7 +          # Color variation
            v_contrast * 0.8 +          # Shadow contrast
            edge_percentage * 0.3       # Edge density (weakest for aerial views)
        )
        
        return combined_score, masked_edges
    
    def adjust_space_coordinates(self, space, frame_width, frame_height):
        """Adjust parking space coordinates based on current frame dimensions"""
        adjusted_points = []
        
        # Step 1: Scale back up if spaces were defined in a scaled-down view
        if self.display_scale != 1.0:
            inverse_scale = 1.0 / self.display_scale
            for x, y in space:
                scaled_x = int(x * inverse_scale)
                scaled_y = int(y * inverse_scale)
                adjusted_points.append((scaled_x, scaled_y))
        else:
            adjusted_points = space.copy()
        
        # Step 2: Adjust if current frame is different size than original
        if hasattr(self, 'original_width') and hasattr(self, 'original_height'):
            if self.original_width > 0 and self.original_height > 0:
                if frame_width != self.original_width or frame_height != self.original_height:
                    # Calculate ratio between current and original dimensions
                    width_ratio = frame_width / self.original_width
                    height_ratio = frame_height / self.original_height
                    
                    # Apply ratio to coordinates
                    ratio_adjusted = []
                    for x, y in adjusted_points:
                        ratio_x = int(x * width_ratio)
                        ratio_y = int(y * height_ratio)
                        ratio_adjusted.append((ratio_x, ratio_y))
                    return ratio_adjusted
        
        return adjusted_points
    
    def check_spaces(self, frame, motion_mask=None):
        """Check each parking space to see if it's occupied"""
        free_spaces = 0
        
        # Check if we have spaces defined
        if len(self.spaces) == 0:
            cv2.putText(frame, "No spaces defined", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, 0
        
        # Get current frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Preprocess frame for static analysis
        gray_frame = self.preprocess_frame(frame)
        
        # Check each space
        for i, space in enumerate(self.spaces):
            try:
                # Adjust coordinates for current frame
                adjusted_points = self.adjust_space_coordinates(space, frame_width, frame_height)
                
                # Create a mask for this space
                mask = np.zeros(gray_frame.shape[:2], dtype=np.uint8)
                contour = np.array(adjusted_points).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [contour], 255)
                
                # Detect vehicle presence using static image analysis
                score, _ = self.detect_vehicle_static(gray_frame, mask)
                
                # If we have motion mask, incorporate motion detection
                if motion_mask is not None:
                    # Apply mask to get only motion inside this space
                    masked_motion = cv2.bitwise_and(motion_mask, motion_mask, mask=mask)
                    motion_pixels = cv2.countNonZero(masked_motion)
                    total_pixels = cv2.countNonZero(mask)
                    motion_percentage = (motion_pixels / total_pixels) * 100 if total_pixels > 0 else 0
                    
                    # Add a small weight to the score if there's motion
                    if motion_percentage > 1.0:
                        score += motion_percentage * 0.3
                
                # Determine if space is free based on score
                # Lower score = fewer edges/features = likely empty space
                is_free = score < self.threshold_value
                
                if is_free:
                    color = (0, 255, 0)  # Green for free
                    thickness = 3
                    free_spaces += 1
                else:
                    color = (0, 0, 255)  # Red for occupied
                    thickness = 2
                
                # Draw space outline
                cv2.polylines(frame, [contour], True, color, thickness)
                
                # Show space number and score
                center_x = sum(p[0] for p in adjusted_points) // 4
                center_y = sum(p[1] for p in adjusted_points) // 4
                cv2.putText(frame, f"#{i}: {score:.1f}%", (center_x-30, center_y+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            except Exception as e:
                print(f"Error with space {i}: {e}")
        
        # Show count of free spaces
        cv2.putText(frame, f"Free: {free_spaces}/{len(self.spaces)}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        
        # Show threshold value
        cv2.putText(frame, f"Threshold: {self.threshold_value}%", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        
        # Send notification if it's time (based on cooldown)
        current_time = time.time()
        if current_time - self.last_notification_time > self.notification_cooldown:
            try:
                if free_spaces > 0:
                    notification.notify(
                        title="Parking Available",
                        message=f"{free_spaces} spaces free",
                        timeout=5
                    )
                else:
                    notification.notify(
                        title="Parking Full",
                        message="No spaces available",
                        timeout=5
                    )
                self.last_notification_time = current_time
            except:
                pass  # Ignore notification errors
        
        return frame, free_spaces
    
    def run(self, spaces_file='parking_spaces.yml'):
        """Run the parking space detector"""
        # Make sure we have input source and try to load spaces
        if not self.choose_input_source():
            return
            
        # Try to load parking spaces
        if not self.load_spaces(spaces_file):
            messagebox.showwarning("Warning", "No parking spaces defined. Run setup first.")
            return
            
        print(f"Running detector with {len(self.spaces)} spaces. Press 'q' to quit.")
        
        # Variables for FPS calculation
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Create analysis windows
        cv2.namedWindow("Parking Space Detector")
        cv2.namedWindow("Analysis View")
        
        try:
            while True:
                # Get current frame
                if self.input_source == 'webcam':
                    # Read from webcam
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:  # screen capture
                    # Get screen dimensions
                    root = tk.Tk()
                    screen_width = root.winfo_screenwidth()
                    screen_height = root.winfo_screenheight()
                    root.destroy()
                    
                    # Take screenshot
                    screen = ImageGrab.grab(bbox=(0, 0, screen_width, screen_height))
                    frame = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)
                
                # Preprocess frame - blur but keep color information
                processed = self.preprocess_frame(frame)
                
                # Create grayscale for edge visualization
                gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                
                # Apply motion detection as supplementary info
                motion_mask = self.motion_detector.detect(gray)
                
                # Check spaces and get result image
                result, free_count = self.check_spaces(processed, motion_mask)
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:  # Update FPS every second
                    fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Show FPS
                cv2.putText(result, f"FPS: {fps:.1f}", (20, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                # Show controls
                h = result.shape[0]
                cv2.putText(result, "q: Quit | +/-: Adjust threshold", 
                           (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Resize for display if too large
                h, w = result.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    result = cv2.resize(result, (int(w*scale), int(h*scale)))
                    edges_display = cv2.resize(edges, (int(w*scale), int(h*scale)))
                    motion_mask_display = cv2.resize(motion_mask, (int(w*scale), int(h*scale)))
                else:
                    edges_display = edges
                    motion_mask_display = motion_mask
                
                # Convert to BGR for display
                edges_visual = cv2.cvtColor(edges_display, cv2.COLOR_GRAY2BGR)
                motion_visual = cv2.cvtColor(motion_mask_display, cv2.COLOR_GRAY2BGR)
                
                # Create a combined analysis view
                analysis_view = np.hstack((edges_visual, motion_visual))
                
                # Show results
                cv2.imshow("Parking Space Detector", result)
                cv2.imshow("Analysis View", analysis_view)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('+') or key == ord('='):  # Increase threshold
                    self.threshold_value = min(100, self.threshold_value + 5)
                elif key == ord('-'):  # Decrease threshold
                    self.threshold_value = max(5, self.threshold_value - 5)
                
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Clean up
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
