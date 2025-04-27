"""
Improved Parking Space Detection System - Main Entry Point
Provides both GUI and command-line interfaces to access the parking space picker and detector
"""
import tkinter as tk
from tkinter import messagebox
import argparse
import os
from detector import ParkingSpaceDetector
from picker import ParkingSpacePicker

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Parking Space Detection System")
    
    # Add arguments
    parser.add_argument("--mode", type=str, choices=["setup", "detect"], 
                        help="Operation mode: 'setup' to mark spaces, 'detect' to run detector")
    parser.add_argument("--data", type=str, default="parking_spaces.yml",
                        help="Path to YAML file for parking space coordinates")
    
    return parser.parse_args()

def show_gui():
    """Show the GUI interface for selecting modes"""
    # Create the main window
    root = tk.Tk()
    root.title("Parking Space Detection")
    root.geometry("320x230")
    root.configure(bg="#f0f0f0")
    
    # Add title label
    tk.Label(
        root, 
        text="Parking Space Detection", 
        font=("Arial", 16, "bold"),
        bg="#f0f0f0"
    ).pack(pady=(20, 10))
    
    # Track the data file for spaces
    data_file = "parking_spaces.yml"
    
    # Define button functions
    def run_setup():
        root.destroy()  # Close main window
        picker = ParkingSpacePicker()
        picker.run()
    
    def run_detector():
        root.destroy()  # Close main window
        detector = ParkingSpaceDetector()
        detector.run(data_file)
        
    def show_help():
        messagebox.showinfo("Help", 
            "1. Setup Parking Spaces:\n"
            "   • Left-click: Add corner points (4 per space)\n"
            "   • Right-click: Remove spaces or clear points\n"
            "   • Press 'q' to save and quit\n\n"
            "2. Run Detector:\n"
            "   • Green: Free space\n"
            "   • Red: Occupied space\n"
            "   • +/-: Adjust detection threshold\n"
            "   • Press 'q' to quit")
    
    # Common button style
    button_style = {"font": ("Arial", 12), "width": 18, "padx": 5, "pady": 5}
    
    # Create buttons
    tk.Button(
        root, text="Setup Parking Spaces", command=run_setup,
        bg="#2196F3", fg="white", **button_style
    ).pack(pady=5)
    
    tk.Button(
        root, text="Run Detector", command=run_detector,
        bg="#4CAF50", fg="white", **button_style
    ).pack(pady=5)
    
    tk.Button(
        root, text="Help", command=show_help,
        bg="#FFC107", fg="black", **button_style
    ).pack(pady=5)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    # Start the application
    root.mainloop()

def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()
    
    # If mode specified in command line, run directly
    if args.mode:
        if args.mode == "setup":
            picker = ParkingSpacePicker()
            picker.run()
        elif args.mode == "detect":
            detector = ParkingSpaceDetector()
            detector.run(args.data)
    else:
        # No command line arguments, show GUI
        show_gui()

# Run the main function when script is executed directly
if __name__ == "__main__":
    main()
