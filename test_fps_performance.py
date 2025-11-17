#!/usr/bin/env python3
"""
FPS Performance Test Script for PiVision

This script helps verify the FPS improvements by testing different configuration settings.
It doesn't require a GUI and outputs FPS measurements to the console.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys

def test_fps_configuration(config_name, width, height, skip_frames, detection_conf, tracking_conf, show_landmarks, duration=10):
    """
    Test FPS with a specific configuration
    
    Args:
        config_name: Name of the configuration being tested
        width: Camera width
        height: Camera height
        skip_frames: Number of frames to skip (0 = no skip)
        detection_conf: MediaPipe detection confidence
        tracking_conf: MediaPipe tracking confidence
        show_landmarks: Whether to draw landmarks
        duration: Test duration in seconds
    """
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Skip Frames: {skip_frames}")
    print(f"  Detection Confidence: {detection_conf}")
    print(f"  Tracking Confidence: {tracking_conf}")
    print(f"  Show Landmarks: {show_landmarks}")
    print(f"{'='*60}")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf
    )
    mp_draw = mp.solutions.drawing_utils
    
    # FPS tracking
    frame_count = 0
    processed_frames = 0
    start_time = time.time()
    
    print(f"\nRunning test for {duration} seconds...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        # Stop after duration
        if elapsed >= duration:
            break
        
        # Frame skipping logic
        should_process = (frame_count % (skip_frames + 1)) == 0
        
        if should_process:
            processed_frames += 1
            
            # Flip and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = mp_hands.process(rgb_frame)
            
            # Draw landmarks if enabled
            if show_landmarks and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    
    # Calculate results
    total_time = time.time() - start_time
    total_fps = frame_count / total_time
    processing_fps = processed_frames / total_time
    
    cap.release()
    mp_hands.close()
    
    print(f"\nResults:")
    print(f"  Total Frames Captured: {frame_count}")
    print(f"  Frames Processed: {processed_frames}")
    print(f"  Total FPS (camera): {total_fps:.2f}")
    print(f"  Processing FPS: {processing_fps:.2f}")
    print(f"  Speedup from skipping: {total_fps/processing_fps:.2f}x" if skip_frames > 0 else "  No frame skipping")
    
    return {
        'config': config_name,
        'total_fps': total_fps,
        'processing_fps': processing_fps,
        'frames_captured': frame_count,
        'frames_processed': processed_frames
    }

def main():
    print("PiVision FPS Performance Test")
    print("="*60)
    
    # Check if camera is available
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        print("ERROR: Could not open camera!")
        sys.exit(1)
    test_cap.release()
    
    print("\nThis test will measure FPS with different configurations.")
    print("Keep your hand in view during testing for realistic results.\n")
    
    test_duration = 10  # seconds per test
    
    # Test configurations
    configs = [
        # (name, width, height, skip_frames, det_conf, track_conf, show_landmarks)
        ("High Quality (Old Default)", 640, 480, 0, 0.75, 0.75, True),
        ("Balanced (New Default)", 320, 240, 0, 0.5, 0.5, True),
        ("Maximum FPS", 320, 240, 1, 0.3, 0.3, False),
        ("Low Resolution", 160, 120, 0, 0.5, 0.5, False),
        ("Aggressive Skipping", 320, 240, 2, 0.5, 0.5, False),
    ]
    
    results = []
    
    for i, (name, w, h, skip, det, track, landmarks) in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Preparing to test: {name}")
        input("Press ENTER to start this test...")
        
        result = test_fps_configuration(name, w, h, skip, det, track, landmarks, test_duration)
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL TESTS")
    print("="*60)
    print(f"{'Configuration':<30} {'Processing FPS':<15} {'Total FPS':<15}")
    print("-"*60)
    
    baseline_fps = results[0]['processing_fps'] if results else 1
    
    for r in results:
        improvement = (r['processing_fps'] / baseline_fps) * 100
        print(f"{r['config']:<30} {r['processing_fps']:>6.2f} FPS      {r['total_fps']:>6.2f} FPS")
        if r != results[0]:
            print(f"  → {improvement:.0f}% of baseline ({(r['processing_fps'] / baseline_fps):.2f}x)")
    
    print("\n" + "="*60)
    print("Test complete! Use these results to choose your optimal settings.")
    print("="*60)

if __name__ == "__main__":
    main()
