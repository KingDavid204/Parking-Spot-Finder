"""
Improved Parking Space Picker
Allows users to define polygon-based parking spaces on images or screen captures
"""
import cv2
import numpy as np
import yaml
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import ImageGrab

class ParkingSpace:
    """A single parking space defined by 4 corner points"""
    
    def __init__(self, points=None):
        # Initialize with empty list if no points provided
        self.points = points if points else []
    
    def add_point(self, x, y):
        """Add a corner point to this space"""
        self.points.append((x, y))
    
    def is_complete(self):
        """Check if all 4 corner points have been defined"""
        return len(self.points) == 4
    
    def contains_point(self, x, y):
        """Check if a point is inside this space"""
        if len(self.points) < 3:  # Need at least 3 points to form a polygon
            return False
            
        # Convert points to a contour and use OpenCV to check if point is inside
        contour = np.array(self.points).reshape((-1, 1, 2)).astype(np.int32)
        return cv2.pointPolygonTest(contour, (x, y), False) >= 0

class ParkingSpacePicker:
    """UI for defining parking spaces on an image"""
    
    def __init__(self):
        # Initialize properties
        self.img = None                  # Display image
        self.original_img = None         # Original unresized image
        self.spaces = []                 # List of parking spaces
        self.current_space = None        # Index of selected space
        self.current_points = []         # Points for space being created
        self.original_width = 0          # Original image width
        self.original_height = 0         # Original image height
        self.display_scale = 1.0         # Display scaling factor
        self.input_source = None         # Input source type
        
    def save_spaces(self, filename='parking_spaces.yml'):
        """Save spaces to YAML file with scaling information"""
        try:
            # Convert spaces to serializable format
            serialized_spaces = []
            for space in self.spaces:
                # Convert tuples to lists for YAML compatibility
                points_as_lists = [list(point) for point in space.points]
                serialized_spaces.append({
                    'points': points_as_lists  # Save as lists, not tuples
                })
                
            # Create data dictionary
            data = {
                'spaces': serialized_spaces,
                'image_dimensions': {
                    'width': self.original_width,
                    'height': self.original_height,
                    'scale': self.display_scale
                }
            }
            
            # Save to YAML file
            with open(filename, 'w') as f:
                yaml.dump(data, f)
                
            print(f"Saved {len(self.spaces)} spaces to {filename}")
            return True
        except Exception as e:
            print(f"Error saving spaces: {e}")
            return False
    
    def load_spaces(self, filename='parking_spaces.yml'):
        """Load saved parking spaces from YAML file"""
        try:
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
                
            # Load dimensions data
            dimensions = data.get('image_dimensions', {})
            self.original_width = dimensions.get('width', 0)
            self.original_height = dimensions.get('height', 0)
            self.display_scale = dimensions.get('scale', 1.0)
            
            # Load spaces
            self.spaces = []
            for space_data in data.get('spaces', []):
                points_lists = space_data.get('points', [])
                # Convert lists back to tuples for internal use
                points_tuples = [tuple(point) for point in points_lists]
                space = ParkingSpace(points_tuples)
                self.spaces.append(space)
                
            print(f"Loaded {len(self.spaces)} spaces from {filename}")
            return True
        except Exception as e:
            print(f"Error loading spaces: {e}")
            self.spaces = []
            return False
    
    def choose_image_source(self):
        """Show dialog to let user select image source"""
        # Create dialog window
        root = tk.Tk()
        root.title("Choose Image")
        root.geometry("300x220")
        root.configure(bg="#f0f0f0")
        
        # Add title
        tk.Label(root, text="Choose Image Source", font=("Arial", 14), bg="#f0f0f0").pack(pady=15)
        
        # Button callbacks
        def select_webcam():
            self.input_source = 'webcam'
            root.destroy()
        
        def select_screen():
            self.input_source = 'screen'
            root.destroy()
            
        def select_file():
            self.input_source = 'file'
            root.destroy()
        
        # Create buttons
        tk.Button(root, text="Webcam", command=select_webcam, 
                  bg="#4CAF50", fg="white", width=12, font=("Arial", 12)).pack(pady=8)
        tk.Button(root, text="Screen Capture", command=select_screen, 
                  bg="#2196F3", fg="white", width=12, font=("Arial", 12)).pack(pady=8)
        tk.Button(root, text="Image File", command=select_file, 
                  bg="#9C27B0", fg="white", width=12, font=("Arial", 12)).pack(pady=8)
        
        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() - root.winfo_width()) // 2
        y = (root.winfo_screenheight() - root.winfo_height()) // 2
        root.geometry(f"+{x}+{y}")
        
        # Show dialog and wait for selection
        root.mainloop()
        
        return self.input_source is not None
    
    def get_image(self):
        """Get image from the selected source"""
        try:
            # Option 1: Webcam capture
            if self.input_source == 'webcam':
                # Open webcam
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    messagebox.showerror("Error", "Could not open webcam")
                    return False
                
                # Show webcam preview until user presses space or esc
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        cap.release()
                        return False
                    
                    # Show instructions on preview
                    cv2.putText(frame, "SPACE to capture, ESC to cancel", (20, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Webcam", frame)
                    
                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        cap.release()
                        cv2.destroyAllWindows()
                        return False
                    elif key == 32:  # SPACE key
                        # Capture frame and store dimensions
                        self.img = frame.copy()
                        self.original_img = self.img.copy()
                        h, w = self.img.shape[:2]
                        self.original_width = w
                        self.original_height = h
                        self.display_scale = 1.0
                        break
                
                # Clean up webcam
                cap.release()
                cv2.destroyAllWindows()
            
            # Option 2: Screen capture
            elif self.input_source == 'screen':
                # Get screen dimensions
                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
                
                # Store original dimensions
                self.original_width = screen_width
                self.original_height = screen_height
                
                # Take screenshot
                screen = ImageGrab.grab(bbox=(0, 0, screen_width, screen_height))
                screen_np = np.array(screen)
                self.img = cv2.cvtColor(screen_np, cv2.COLOR_RGB2BGR)
                
                # Store original image
                self.original_img = self.img.copy()
                
                # Resize for display if needed
                h, w = self.img.shape[:2]
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    self.display_scale = scale
                    self.img = cv2.resize(self.img, (int(w*scale), int(h*scale)))
                else:
                    self.display_scale = 1.0
            
            # Option 3: Load from file
            elif self.input_source == 'file':
                # Open file dialog
                file_path = filedialog.askopenfilename(
                    title="Select Image",
                    filetypes=[("Image files", "*.jpg *.jpeg *.png")]
                )
                
                if not file_path:
                    return False
                
                # Load image
                self.img = cv2.imread(file_path)
                if self.img is None:
                    messagebox.showerror("Error", "Could not load image")
                    return False
                
                # Store original
                self.original_img = self.img.copy()
                h, w = self.img.shape[:2]
                self.original_width = w
                self.original_height = h
                
                # Resize for display if needed
                if w > 1200 or h > 800:
                    scale = min(1200/w, 800/h)
                    self.display_scale = scale
                    self.img = cv2.resize(self.img, (int(w*scale), int(h*scale)))
                else:
                    self.display_scale = 1.0
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Image capture error: {e}")
            return False
    
    def mouse_events(self, event, x, y, flags, param):
        """Handle mouse interaction for creating and selecting spaces"""
        # Left button pressed - add point or select existing space
        if event == cv2.EVENT_LBUTTONDOWN:
            # First check if clicking on an existing space
            for i, space in enumerate(self.spaces):
                if space.contains_point(x, y):
                    self.current_space = i  # Select this space
                    return
            
            # If not on existing space, add point to current space being created
            if len(self.current_points) < 4:
                self.current_points.append((x, y))
                
                # If we've added 4 points, create a new space
                if len(self.current_points) == 4:
                    space = ParkingSpace(self.current_points)
                    self.spaces.append(space)
                    self.current_space = len(self.spaces) - 1
                    self.current_points = []  # Reset for next space
                    self.save_spaces()
        
        # Right button pressed - delete space or clear current points
        elif event == cv2.EVENT_RBUTTONDOWN:
            # If we have points being added, clear them
            if self.current_points:
                self.current_points = []
                return
                
            # Otherwise try to delete an existing space
            for i, space in enumerate(self.spaces):
                if space.contains_point(x, y):
                    # Remove this space
                    self.spaces.pop(i)
                    self.current_space = None
                    self.save_spaces()
                    break
    
    def run(self):
        """Run the parking space picker interface"""
        # Get image from selected source
        if not self.choose_image_source() or not self.get_image():
            return
        
        # Show instructions
        messagebox.showinfo("Instructions", 
            "• Left-click: Add corner point (4 points per space)\n"
            "• Right-click: Remove a space or clear current points\n"
            "• Click on a space: Select it\n"
            "• q: Save and quit\n"
            "• r: Reset all spaces")
        
        # Create window and register mouse callback
        cv2.namedWindow("Parking Space Picker")
        cv2.setMouseCallback("Parking Space Picker", self.mouse_events)
        
        # Main loop
        try:
            while True:
                # Create copy of image to draw on
                display = self.img.copy()
                
                # Draw all existing spaces
                for i, space in enumerate(self.spaces):
                    # Get points for this space
                    points = np.array(space.points).reshape((-1, 1, 2)).astype(np.int32)
                    
                    # Set color (yellow for selected, purple for others)
                    color = (0, 255, 255) if i == self.current_space else (255, 0, 255)
                    thickness = 3 if i == self.current_space else 2
                    
                    # Draw the space outline
                    cv2.polylines(display, [points], True, color, thickness)
                    
                    # Add space number
                    center_x = sum(p[0] for p in space.points) // 4
                    center_y = sum(p[1] for p in space.points) // 4
                    cv2.putText(display, f"#{i}", (center_x-10, center_y+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw points being added for current space
                for i, point in enumerate(self.current_points):
                    # Draw point with number
                    cv2.circle(display, point, 5, (0, 255, 0), -1)
                    cv2.putText(display, str(i+1), (point[0]+5, point[1]+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw lines between points
                if len(self.current_points) > 1:
                    for i in range(len(self.current_points)):
                        cv2.line(display, 
                                self.current_points[i], 
                                self.current_points[(i+1) % len(self.current_points)], 
                                (0, 255, 0), 2)
                
                # Show count of spaces
                cv2.putText(display, f"Parking Spaces: {len(self.spaces)}", (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show number of points added for current space
                if self.current_points:
                    cv2.putText(display, f"Adding Space: {len(self.current_points)}/4 points", 
                              (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show scaling information if image is scaled
                if self.display_scale < 1.0:
                    cv2.putText(display, f"Display scale: {self.display_scale:.2f}", (20, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show controls at bottom
                h = display.shape[0]
                cv2.putText(display, "Left-click: Add point | Right-click: Remove | q: Quit | r: Reset", 
                           (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show image
                cv2.imshow("Parking Space Picker", display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    break
                elif key == ord('r'):  # Reset all spaces
                    if messagebox.askyesno("Reset", "Remove all parking spaces?"):
                        self.spaces = []
                        self.current_space = None
                        self.current_points = []
                        self.save_spaces()
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Clean up
            cv2.destroyAllWindows()
            print(f"Setup complete. Saved {len(self.spaces)} parking spaces.")
