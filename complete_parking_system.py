import cv2
import os
import sys
import subprocess
import json
import numpy as np
from pathlib import Path

# ============================================
# PACKAGE INSTALLATION CHECK
# ============================================

def check_and_install_packages():
    """Check if required packages are installed, install if missing"""
    required_packages = {
        'ultralytics': 'ultralytics',
        'shapely': 'shapely',
        'lap': 'lap'
    }
    
    missing_packages = []
    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"OK: {package} already installed")
        except ImportError:
            print(f"Missing: {package} not found, will install...")
            missing_packages.append(install_name)
    
    if missing_packages:
        print("\nInstalling missing packages...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"OK: Installed {package}")
            except:
                print(f"Failed: Failed to install {package}")
        
        print("\nOK: Installation complete! Please restart the program.")
        input("Press ENTER to exit...")
        sys.exit()

# Run package check
print("Checking required packages...")
check_and_install_packages()

# Now import
from ultralytics import YOLO

# ============================================
# CONFIGURATION - EDIT THESE PATHS IF NEEDED
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "easy.mp4")
FRAME_PATH = os.path.join(BASE_DIR, "parking_frame.jpg")
JSON_OUTPUT_PATH = os.path.join(BASE_DIR, "parking_slots.json")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "parking_output.mp4")

# ============================================
# FUNCTION: Extract frame from video
# ============================================

def extract_frame():
    """Extract first frame from video for drawing slots"""
    print("\n" + "=" * 50)
    print("EXTRACTING FRAME FROM VIDEO")
    print("=" * 50)
    
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        print("Please check the video path in CONFIGURATION section")
        return False
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(FRAME_PATH, frame)
        print(f"OK: Frame saved to: {FRAME_PATH}")
        return True
    else:
        print("ERROR: Could not read first frame from video")
        return False

# ============================================
# FUNCTION: Draw parking slots (Click 4 points)
# ============================================

def draw_slots():
    """Draw parking slots by clicking 4 points per slot"""
    print("\n" + "=" * 60)
    print("DRAW PARKING SLOTS - CLICK 4 POINTS PER SLOT")
    print("=" * 60)
    
    # Get frame
    if not os.path.exists(FRAME_PATH):
        print("No frame found. Extracting first...")
        if not extract_frame():
            return False
    
    # Load image
    img = cv2.imread(FRAME_PATH)
    if img is None:
        print("ERROR: Could not load frame")
        return False
    
    img_copy = img.copy()
    slots = []
    current_points = []
    slot_count = 0
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        nonlocal current_points, img_copy, slot_count, slots
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            current_points.append([x, y])
            
            # Draw point
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
            
            # Draw line from previous point
            if len(current_points) > 1:
                cv2.line(img_copy, 
                        tuple(current_points[-2]), 
                        tuple(current_points[-1]), 
                        (0, 255, 0), 2)
            
            # Complete slot after 4 points
            if len(current_points) == 4:
                # Close polygon (connect last point to first)
                cv2.line(img_copy, 
                        tuple(current_points[-1]), 
                        tuple(current_points[0]), 
                        (0, 255, 0), 2)
                
                # Add slot to list
                slots.append(current_points.copy())
                
                # Add slot number
                cx = sum(p[0] for p in current_points) // 4
                cy = sum(p[1] for p in current_points) // 4
                cv2.putText(img_copy, str(slot_count), (cx-10, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                print(f"OK: Slot {slot_count} completed")
                slot_count += 1
                current_points = []  # Reset for next slot
            
            print(f"Point {len(current_points)}/4 at ({x}, {y})")
            cv2.imshow("Draw Parking Slots (Click 4 points)", img_copy)
    
    # Create window and set mouse callback
    cv2.namedWindow("Draw Parking Slots (Click 4 points)")
    cv2.setMouseCallback("Draw Parking Slots (Click 4 points)", mouse_callback)
    
    print("\nINSTRUCTIONS:")
    print("-" * 40)
    print("1. For EACH parking slot, click 4 corners in ORDER:")
    print("   • Point 1: Top-left corner")
    print("   • Point 2: Top-right corner")
    print("   • Point 3: Bottom-right corner")
    print("   • Point 4: Bottom-left corner")

    print("2. Slot will auto-complete after 4 clicks")
    print("3. Draw ALL parking slots in the image")
    print("4. Press 's' to SAVE and quit")
    print("5. Press 'c' to CLEAR all slots")
    print("6. Press 'q' to QUIT without saving")
    print("-" * 40)
    print(f"Image size: {img.shape[1]} x {img.shape[0]} pixels")
    print("=" * 60)
    
    cv2.imshow("Draw Parking Slots (Click 4 points)", img_copy)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # Save
            if len(slots) > 0:
                # Prepare data in required format
                data = {"slots": []}
                for i, slot_points in enumerate(slots):
                    data["slots"].append({
                        "id": i,
                        "points": slot_points
                    })
                
                # Save to JSON
                with open(JSON_OUTPUT_PATH, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"\nSUCCESS! Saved {len(slots)} parking slots")
                print(f"   Location: {JSON_OUTPUT_PATH}")
                
                # Show final preview
                final_img = img.copy()
                for i, slot_points in enumerate(slots):
                    pts = np.array(slot_points, np.int32)
                    cv2.polylines(final_img, [pts], True, (0, 255, 0), 2)
                    cx = sum(p[0] for p in slot_points) // 4
                    cy = sum(p[1] for p in slot_points) // 4
                    cv2.putText(final_img, str(i), (cx-10, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                cv2.putText(final_img, f"Total Slots: {len(slots)}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("Final Parking Slots - Press any key", final_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                return True
            else:
                print("\nWarning: No slots drawn! Draw at least one slot.")
        
        elif key == ord('c'):  # Clear all
            slots = []
            current_points = []
            slot_count = 0
            img_copy = img.copy()
            cv2.imshow("Draw Parking Slots (Click 4 points)", img_copy)
            print("\nCleared all slots")
        
        elif key == ord('q'):  # Quit without saving
            cv2.destroyAllWindows()
            print("\nQuit without saving")
            return False
    
    return False

# ============================================
# FUNCTION: Run parking detection
# ============================================

def run_detection():
    """Run parking slot detection on video"""
    print("\n" + "=" * 50)
    print("RUNNING PARKING DETECTION")
    print("=" * 50)
    
    # Check if required files exist
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        return
    
    if not os.path.exists(JSON_OUTPUT_PATH):
        print(f"ERROR: Slots file not found: {JSON_OUTPUT_PATH}")
        print("Please draw slots first (Option 2)")
        return
    
    # Load parking slots
    try:
        with open(JSON_OUTPUT_PATH, 'r') as f:
            data = json.load(f)
            slots = data['slots']
        print(f"OK: Loaded {len(slots)} parking slots")
    except Exception as e:
        print(f"ERROR: Error loading slots file: {e}")
        return
    
    # Load YOLO model
    print("\nLoading YOLO model...")
    try:
        model = YOLO('yolo11n.pt')
        print("OK: Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Error loading model: {e}")
        return
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open video")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height}, {fps}fps, {total_frames} frames")
    
    # Setup video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    print("\nProcessing... Press 'q' to stop")
    print("-" * 40)
    
    # Point-in-polygon function (check if a point is inside a polygon)
    def point_in_polygon(point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    frame_count = 0
    cv2.namedWindow('Parking Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Parking Detection', 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect cars in frame
        results = model(frame, conf=0.4, verbose=False)
        
        # Get car center points
        car_centers = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls[0]) == 2:  # Class 2 is 'car' in COCO dataset
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    car_centers.append(center)
                    
                    # Draw car bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        
        # Check each parking slot
        free_count = 0
        occupied_count = 0
        
        for i, slot in enumerate(slots):
            points = np.array(slot['points'], np.int32)
            
            # Check if any car center is inside this slot
            is_occupied = False
            for center in car_centers:
                if point_in_polygon(center, slot['points']):
                    is_occupied = True
                    break
            
            # Set color based on occupancy
            if is_occupied:
                color = (0, 0, 255)  # Red for occupied
                status = "OCC"
                occupied_count += 1
            else:
                color = (0, 255, 0)  # Green for free
                status = "FREE"
                free_count += 1
            
            # Draw parking slot
            cv2.polylines(frame, [points], True, color, 3)
            
            # Add slot label
            cx = sum(p[0] for p in slot['points']) // 4
            cy = sum(p[1] for p in slot['points']) // 4
            cv2.putText(frame, f"{i}:{status}", (cx-30, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display statistics
        stats = f"FREE: {free_count}  OCCUPIED: {occupied_count}  TOTAL: {len(slots)}"
        cv2.putText(frame, stats, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display frame counter
        frame_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame, frame_text, (width-200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow('Parking Detection', frame)
        
        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}%", end='\r')
        
        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n\nDetection complete!")
    print(f"Output saved to: {OUTPUT_VIDEO_PATH}")

# ============================================
# FUNCTION: Check system status
# ============================================

def check_system():
    """Check if all required files exist"""
    print("\n" + "=" * 50)
    print("SYSTEM CHECK")
    print("=" * 50)
    
    # Check video file
    print(f"\nVideo file:")
    if os.path.exists(VIDEO_PATH):
        size = os.path.getsize(VIDEO_PATH) / (1024*1024)
        print(f"   OK: Found ({size:.1f} MB)")
    else:
        print(f"   ERROR: Not found: {VIDEO_PATH}")
    
    # Check frame image
    print(f"\nFrame image:")
    if os.path.exists(FRAME_PATH):
        size = os.path.getsize(FRAME_PATH) / 1024
        print(f"   OK: Found ({size:.1f} KB)")
    else:
        print(f"   ERROR: Not found (run Option 1)")
    
    # Check slots JSON
    print(f"\nSlots file:")
    if os.path.exists(JSON_OUTPUT_PATH):
        size = os.path.getsize(JSON_OUTPUT_PATH)
        print(f"   OK: Found ({size} bytes)")
    else:
        print(f"   ERROR: Not found (run Option 2)")
    
    print("\n" + "=" * 50)

# ============================================
# MAIN MENU
# ============================================

def main():
    while True:
        print("\n" + "="*60)
        print("COMPLETE PARKING SLOT DETECTION SYSTEM")
        print("="*60)
        print("1. Extract Frame from Video")
        print("2. Draw Parking Slots (Click 4 points per slot)")
        print("3. Run Detection on Video")
        print("4. Check System Status")
        print("5. Exit")
        print("-"*60)
        print("Follow this order: 1 -> 2 -> 3")
        print("="*60)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            extract_frame()
            input("\nPress ENTER to continue...")
        
        elif choice == '2':
            if draw_slots():
                print("\nOK: Slots saved! You can now run detection (Option 3)")
            input("\nPress ENTER to continue...")
        
        elif choice == '3':
            run_detection()
            input("\nPress ENTER to continue...")
        
        elif choice == '4':
            check_system()
            input("\nPress ENTER to continue...")
        
        elif choice == '5':
            print("\nThank you for using Parking Slot Detection System!")
            print("Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number between 1 and 5.")
            input("\nPress ENTER to continue...")

# ============================================
# PROGRAM ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()