import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
cap = cv2.VideoCapture(0)

# Smoothing parameters
prev_loc_x, prev_loc_y = 0, 0
curr_loc_x, curr_loc_y = 0, 0
smooth_factor = 5  # higher = smoother

click_delay = 0.5
last_left_click_time = 0
last_right_click_time = 0

drag_mode = False
prev_time = 0

# Overlay text
def draw_overlay(image):
    cv2.rectangle(image, (5, 5), (330, 110), (50, 50, 50), -1)
    cv2.putText(image, 'Gesture Guide:', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, 'Index + Thumb     : Left Click', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
    cv2.putText(image, 'Middle + Thumb    : Right Click', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(image, 'Index + Middle     : Drag & Drop', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 200), 1)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.7) as hands:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        image_height, image_width, _ = frame.shape

        # FPS Limiting (~30fps)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        if fps > 35:
            time.sleep(0.01)

        # Process hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw overlay
        draw_overlay(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                index = landmarks[8]
                thumb = landmarks[4]
                middle = landmarks[12]

                x = int(index.x * image_width)
                y = int(index.y * image_height)

                # Screen mapping with interpolation
                target_x = np.interp(x, [0, image_width], [0, screen_width])
                target_y = np.interp(y, [0, image_height], [0, screen_height])

                # Exponential smoothing
                curr_loc_x = prev_loc_x + (target_x - prev_loc_x) / smooth_factor
                curr_loc_y = prev_loc_y + (target_y - prev_loc_y) / smooth_factor
                pyautogui.moveTo(curr_loc_x, curr_loc_y)
                prev_loc_x, prev_loc_y = curr_loc_x, curr_loc_y

                # Finger distances
                index_thumb_dist = np.hypot(index.x - thumb.x, index.y - thumb.y)
                middle_thumb_dist = np.hypot(middle.x - thumb.x, middle.y - thumb.y)
                index_middle_dist = np.hypot(index.x - middle.x, index.y - middle.y)

                now = time.time()

                # LEFT CLICK
                if index_thumb_dist < 0.03 and now - last_left_click_time > click_delay:
                    pyautogui.click()
                    last_left_click_time = now
                    cv2.circle(frame, (x, y), 15, (0, 255, 0), -1)

                # RIGHT CLICK
                elif middle_thumb_dist < 0.03 and now - last_right_click_time > click_delay:
                    pyautogui.click(button='right')
                    last_right_click_time = now
                    cv2.circle(frame, (x, y), 15, (255, 0, 0), -1)

                # DRAG & DROP toggle
                if index_middle_dist < 0.04:
                    if not drag_mode:
                        pyautogui.mouseDown()
                        drag_mode = True
                        cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)
                else:
                    if drag_mode:
                        pyautogui.mouseUp()
                        drag_mode = False

                # Draw hand
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show webcam window
        cv2.imshow("AI Virtual Mouse - Smooth Mode", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
