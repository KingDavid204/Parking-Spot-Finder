# Parking-Spot-Finder
An intelligent solution designed to optimize parking management using real-time computer vision.

This is an enhanced parking space detection system inspired by the OlgaRose ParkingLot project. It uses OpenCV to detect free parking spaces in images or video feeds.

## Features

- **Polygon-based parking spaces**: Define parking spaces by clicking on their exact corners, not just rectangles
- **Background subtraction**: Uses advanced motion detection to identify occupied spaces
- **YAML configuration**: Stores parking space coordinates in readable YAML format
- **Command-line interface**: Run directly from the command line or use the GUI
- **Adjustable detection**: Fine-tune detection sensitivity with threshold controls

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- PyYAML
- tkinter
- PIL (Pillow)
- plyer (for notifications)

You can install the required packages with:

```
pip install opencv-python numpy pyyaml pillow plyer
```

## Usage

### GUI Mode

Simply run:

```
python main.py
```

This will open a GUI where you can choose to set up parking spaces or run the detector.

### Command Line Mode

You can also run the application directly from the command line:

```
# To set up parking spaces
python main.py --mode setup

# To run the detector
python main.py --mode detect --data parking_spaces.yml
```

## How It Works

1. **Setup Phase**
   - Choose an image source (webcam, screen capture, or image file)
   - Click on the 4 corners of each parking space you want to monitor
   - Spaces are saved in a YAML file for reuse

2. **Detection Phase**
   - The system loads your defined parking spaces
   - Uses background subtraction to detect motion
   - Analyzes each space to determine if it's occupied or free
   - Displays results in real-time with color-coded overlays

## Controls

### Setup Mode
- **Left-click**: Add a corner point (4 points per space)
- **Right-click**: Remove a space or clear current points
- **q**: Save and quit
- **r**: Reset all spaces

### Detector Mode
- **+/-**: Adjust detection threshold
- **q**: Quit

## Improvements Over Original Implementation

- More accurate polygon-based space definition
- Better detection using background subtraction instead of simple edge detection
- More flexible configuration with YAML instead of pickle files
- Command-line interface for automation
- Adjustable detection sensitivity
