# PiVision Performance Optimization Guide

## Overview

PiVision now includes several performance optimization settings to significantly increase the FPS (frames per second) of gesture recognition. This guide explains how to use these settings to achieve the best performance for your system.

## Performance Settings

Access these settings by clicking **⚙ Settings** from the home screen.

### Camera Resolution

- **Camera Width**: Default 320 (range: 160-1920)
- **Camera Height**: Default 240 (range: 120-1080)

**Impact**: Lower resolution = Higher FPS
- **Recommended for high FPS**: 320x240 or 160x120
- **Recommended for accuracy**: 640x480
- **Note**: Settings take effect when you restart gesture recognition

### Frame Skipping

- **Skip Frames**: Default 0 (range: 0-5)
  - 0 = Process every frame (no skipping)
  - 1 = Process every other frame (2x speed improvement)
  - 2 = Process every third frame (3x speed improvement)
  - etc.

**Impact**: Higher skip value = Higher FPS but slightly delayed gesture detection
- **Recommended for real-time control**: 0-1
- **Recommended for maximum FPS**: 2-3

### Visual Overlays

- **Show Hand Landmarks**: Yes/No (Default: Yes)

**Impact**: Disabling landmarks = Slight FPS improvement
- **Recommended for maximum FPS**: No
- **Recommended for debugging**: Yes

### MediaPipe Confidence Thresholds

- **Min Detection Confidence**: Default 0.5 (range: 0.1-1.0)
- **Min Tracking Confidence**: Default 0.5 (range: 0.1-1.0)

**Impact**: Lower values = Faster detection but may increase false positives
- **Recommended for high FPS**: 0.3-0.5
- **Recommended for accuracy**: 0.7-0.9

## Recommended Settings Profiles

### Maximum FPS (for fast-paced actions)
```
Camera Width: 320
Camera Height: 240
Skip Frames: 1-2
Show Landmarks: No
Min Detection Confidence: 0.3
Min Tracking Confidence: 0.3
```

### Balanced (default)
```
Camera Width: 320
Camera Height: 240
Skip Frames: 0
Show Landmarks: Yes
Min Detection Confidence: 0.5
Min Tracking Confidence: 0.5
```

### Maximum Accuracy (slower but more precise)
```
Camera Width: 640
Camera Height: 480
Skip Frames: 0
Show Landmarks: Yes
Min Detection Confidence: 0.8
Min Tracking Confidence: 0.8
```

## FPS Monitor

The current FPS is displayed in the bottom-left corner of the video feed in yellow text. Use this to monitor the impact of your settings changes.

## Tips for Best Performance

1. **Start with defaults**: Begin with the balanced settings and adjust based on your needs
2. **Monitor FPS**: Keep an eye on the FPS counter to see real-time impact
3. **Test incrementally**: Change one setting at a time to understand its impact
4. **System-specific**: Performance will vary based on your camera, CPU, and GPU
5. **GPU Acceleration**: The application automatically uses DirectML (Windows) or CUDA (NVIDIA) if available for faster inference

## Troubleshooting Low FPS

If you're still experiencing low FPS after optimization:

1. Close other resource-intensive applications
2. Ensure your camera drivers are up to date
3. Check if GPU acceleration is being used (check console output)
4. Try reducing camera resolution to the minimum (160x120)
5. Increase frame skipping to 2 or 3
6. Disable landmark drawing

## Technical Details

### What affects FPS?

1. **Camera resolution**: Higher resolution = more pixels to process
2. **MediaPipe hand detection**: Runs on every processed frame
3. **ONNX model inference**: Classifies gestures for each detected hand
4. **Frame rendering**: Drawing landmarks and text overlays
5. **Qt display**: Converting and displaying frames in the GUI

### Optimizations implemented

- Reduced default resolution from 640x480 to 320x240
- Optional frame skipping to reduce processing load
- Configurable MediaPipe confidence thresholds (default reduced from 0.75 to 0.5)
- Optional landmark drawing toggle
- Fast transformation for Qt image scaling
- Reduced camera buffer size for more recent frames
- Real-time FPS counter
