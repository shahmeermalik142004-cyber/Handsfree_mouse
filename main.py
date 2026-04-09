import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import math
import numpy as np
import urllib.request
import os
import time

pyautogui.FAILSAFE = False 

model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe AI Model for face tracking (only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as e:
        exit()

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.7,
    min_face_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector = vision.FaceLandmarker.create_from_options(options)

screen_width, screen_height = pyautogui.size()
cam_width, cam_height = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calc_3d(l1, l2):
    return math.sqrt((l1.x - l2.x)**2 + (l1.y - l2.y)**2 + (l1.z - l2.z)**2)

def nothing(x):
    pass

window_name = "Assistive Tech Overhaul (Press 'Q' to Exit)"
cv2.namedWindow(window_name)

# --- CREATE SETTINGS TRACKBARS ---
cv2.createTrackbar("Max Speed", window_name, 35, 100, nothing)
cv2.createTrackbar("Smoothing", window_name, 5, 20, nothing)
cv2.createTrackbar("Deadzone %", window_name, 15, 40, nothing)
cv2.createTrackbar("Blink Thresh", window_name, 19, 30, nothing)
cv2.createTrackbar("Mouth Active %", window_name, 40, 80, nothing)
cv2.createTrackbar("Pout DblClick %", window_name, 80, 150, nothing) 

blink_start_time = 0
is_blinking = False

wink_start_time = 0
is_winking = False
has_toggled_drag = False
is_dragging = False

is_mouth_physically_open = False
right_click_visual_timer = 0  

is_double_clicking = False

curr_x, curr_y = pyautogui.position()

print("[OK] Enhanced Assistive Mode Initialized!")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = detector.detect(mp_image)

    max_speed = max(1, cv2.getTrackbarPos("Max Speed", window_name))
    smoothing_val = max(1, cv2.getTrackbarPos("Smoothing", window_name))
    deadzone_pct = max(2, cv2.getTrackbarPos("Deadzone %", window_name))
    blink_val = max(10, cv2.getTrackbarPos("Blink Thresh", window_name))
    mouth_val = max(15, cv2.getTrackbarPos("Mouth Active %", window_name))
    pout_val = max(10, cv2.getTrackbarPos("Pout DblClick %", window_name))

    pyautogui.PAUSE = 0.0

    center_x = w // 2
    center_y = h // 2
    deadzone_r = int((deadzone_pct / 100.0) * w)
    
    cv2.circle(img, (center_x, center_y), deadzone_r, (0, 255, 0), 2)
    cv2.circle(img, (center_x, center_y), 3, (0, 255, 0), -1)

    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        
        nose_x = int(landmarks[4].x * w)
        nose_y = int(landmarks[4].y * h)
        cv2.circle(img, (nose_x, nose_y), 5, (0, 255, 255), cv2.FILLED)
        
        cv2.line(img, (center_x, center_y), (nose_x, nose_y), (255, 0, 0), 1)

        distance_from_center = calculate_distance((nose_x, nose_y), (center_x, center_y))
        
        if distance_from_center > deadzone_r:
            overshoot = distance_from_center - deadzone_r
            speed = min(max_speed, overshoot / smoothing_val)
            vel_x = ((nose_x - center_x) / distance_from_center) * speed
            vel_y = ((nose_y - center_y) / distance_from_center) * speed
            
            real_x, real_y = pyautogui.position()
            new_x = real_x + vel_x
            new_y = real_y + vel_y
            
            new_x = np.clip(new_x, 0, screen_width)
            new_y = np.clip(new_y, 0, screen_height)
            try:
                pyautogui.moveTo(new_x, new_y)
            except Exception:
                pass
        
        # --- FACIAL GESTURE CALCULATIONS ---
        
        p159 = (int(landmarks[159].x * w), int(landmarks[159].y * h))
        p145 = (int(landmarks[145].x * w), int(landmarks[145].y * h))
        p33  = (int(landmarks[33].x * w), int(landmarks[33].y * h))
        p133 = (int(landmarks[133].x * w), int(landmarks[133].y * h))
        left_vert = calculate_distance(p159, p145)
        left_horiz = calculate_distance(p33, p133)
        left_ear = left_vert / left_horiz if left_horiz != 0 else 0

        p386 = (int(landmarks[386].x * w), int(landmarks[386].y * h))
        p374 = (int(landmarks[374].x * w), int(landmarks[374].y * h))
        p362 = (int(landmarks[362].x * w), int(landmarks[362].y * h))
        p263 = (int(landmarks[263].x * w), int(landmarks[263].y * h))
        right_vert = calculate_distance(p386, p374)
        right_horiz = calculate_distance(p362, p263)
        right_ear = right_vert / right_horiz if right_horiz != 0 else 0
        
        p13 = (int(landmarks[13].x * w), int(landmarks[13].y * h))
        p14 = (int(landmarks[14].x * w), int(landmarks[14].y * h))
        mouth_dist = calculate_distance(p13, p14)
        
        mouth_width_3d = calc_3d(landmarks[61], landmarks[291]) 
        eyes_3d_dist = calc_3d(landmarks[159], landmarks[386])
        mouth_ratio = (mouth_width_3d / eyes_3d_dist) * 100 if eyes_3d_dist != 0 else 0
        
        p61 = (int(landmarks[61].x * w), int(landmarks[61].y * h)) 
        p291 = (int(landmarks[291].x * w), int(landmarks[291].y * h))
        
        cv2.line(img, p159, p145, (0, 255, 0), 1)
        cv2.line(img, p386, p374, (0, 255, 255), 1)
        cv2.line(img, p13, p14, (255, 255, 255), 1)
        cv2.line(img, p61, p291, (255, 0, 255), 1)

        # --- LOGIC TRIGGERS ---
        
        ear_threshold_limit = blink_val / 100.0  
        
        # 1. INTENTIONAL BLINK FILTER (Left Click) - BOTH EYES
        if left_ear < ear_threshold_limit and right_ear < ear_threshold_limit:
            is_winking = False # Cancel wink tracking if we detect a full double blink
            if not is_blinking:
                is_blinking = True
                blink_start_time = time.time()
                cv2.putText(img, 'Blinking...', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            else:
                blink_duration = time.time() - blink_start_time
                if blink_duration > 0.15: 
                    pyautogui.click()
                    cv2.putText(img, 'LEFT CLICK!', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    blink_start_time = time.time() + 0.5 
                    
        # 2. INTENTIONAL WINK FILTER (Toggle Drag) - LEFT EYE ONLY
        # We ensure right_ear is clearly OPEN (0.05 buffer over threshold) to guarantee it's an isolated wink
        elif left_ear < ear_threshold_limit and right_ear > (ear_threshold_limit + 0.05):
            is_blinking = False # Cancel blink tracking
            if not is_winking:
                is_winking = True
                wink_start_time = time.time()
                has_toggled_drag = False
                cv2.putText(img, 'Winking...', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 100, 0), 3)
            else:
                if not has_toggled_drag:
                    wink_duration = time.time() - wink_start_time
                    if wink_duration > 0.15: 
                        has_toggled_drag = True
                        if is_dragging:
                            is_dragging = False
                            pyautogui.mouseUp()
                        else:
                            is_dragging = True
                            pyautogui.mouseDown()
        else:
            is_blinking = False
            is_winking = False
            
        # 3. RIGHT CLICK (Mouth Open)
        # Simplified since we ripped Drag off it. Just a quick responsive tap!
        if mouth_dist > mouth_val:
            if not is_mouth_physically_open:
                is_mouth_physically_open = True
                pyautogui.click(button='right')
                right_click_visual_timer = time.time() + 0.5 
        else:
            is_mouth_physically_open = False
        
        # 4. DOUBLE CLICK (Pout)
        if mouth_ratio < pout_val:
            if not is_double_clicking:
                is_double_clicking = True
                pyautogui.doubleClick()
                cv2.putText(img, 'DOUBLE CLICK!', (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        else:
            is_double_clicking = False

    # Persistent Visuals
    if is_dragging:
        cv2.putText(img, 'DRAG PINNED! (Wink left eye to drop)', (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.rectangle(img, (10, 10), (w-10, h-10), (0, 255, 255), 5) 
        
    if time.time() < right_click_visual_timer:
        cv2.putText(img, 'RIGHT CLICK!', (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    if detection_result.face_landmarks:
        cv2.putText(img, f'Pout Ratio: {mouth_ratio:.1f} (Need: <{pout_val})', (50, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
