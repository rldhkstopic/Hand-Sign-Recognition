import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
import os
import json
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

save_path = "dataset"
json_path = "dataset/hand_landmarks.json"

if not os.path.exists(save_path):
    os.makedirs(save_path)

Target = [
            {'key' : 'ㄱ', 'eng' : 'giyeok'}, 
            {'key' : 'ㄴ', 'eng' : 'nieun'}, 
            {'key' : 'ㄷ', 'eng' : 'digeut'}, 
            {'key' : 'ㄹ', 'eng' : 'rieul'}, 
            {'key' : 'ㅁ', 'eng' : 'mieum'}, 
            {'key' : 'ㅂ', 'eng' : 'bieup'}, 
            {'key' : 'ㅅ', 'eng' : 'siot'}, 
            {'key' : 'ㅇ', 'eng' : 'ieung'}, 
            {'key' : 'ㅈ', 'eng' : 'jieut'}, 
            {'key' : 'ㅊ', 'eng' : 'chieut'}, 
            {'key' : 'ㅋ', 'eng' : 'kieuk'}, 
            {'key' : 'ㅌ', 'eng' : 'tieut'}, 
            {'key' : 'ㅍ', 'eng' : 'pieup'}, 
            {'key' : 'ㅎ', 'eng' : 'hieut'}, 
            {'key' : 'ㅏ', 'eng' : 'a'}, 
            {'key' : 'ㅑ', 'eng' : 'ya'}, 
            {'key' : 'ㅓ', 'eng' : 'eo'}, 
            {'key' : 'ㅕ', 'eng' : 'yeo'}, 
            {'key' : 'ㅗ', 'eng' : 'o'}, 
            {'key' : 'ㅛ', 'eng' : 'yo'}, 
            {'key' : 'ㅜ', 'eng' : 'u'}, 
            {'key' : 'ㅠ', 'eng' : 'yu'}, 
            {'key' : 'ㅡ', 'eng' : 'eu'}, 
            {'key' : 'ㅣ', 'eng' : 'i'}, 
            {'key' : 'ㅔ', 'eng' : 'e'}, 
            {'key' : 'ㅖ', 'eng' : 'ye'}, 
            {'key' : 'ㅐ', 'eng' : 'ae'}, 
            {'key' : 'ㅡ', 'eng' : 'eu'}, 
            {'key' : 'ㅢ', 'eng' : 'ui'}, 
            {'key' : 'ㅚ', 'eng' : 'oe'}, 
            {'key' : 'ㅟ', 'eng' : 'wi'}, 

        ]

selected_char = Target[0]['eng']

data = []
index = 1
if os.path.exists(json_path):
    try:
        with open(json_path, "r") as file:
            data = json.load(file)
            if data:
                index = max(entry["index"] for entry in data) + 1
    except json.JSONDecodeError:
        print(f"Error reading {json_path}. File might be empty or corrupt. Initializing a new dataset.")
else:
    print(f"{json_path} does not exist. Initializing a new dataset.")


try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.flip(color_image, 1)

        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key == 13:  # 엔터
            if results.multi_hand_landmarks:
                filename = os.path.join(save_path, f"{selected_char}_{index}.jpg")
                success = cv2.imwrite(filename, color_image)
                if not success:
                    print(f"Failed to save image at {filename}")


                entry = {
                    "label": selected_char,
                    "index": index
                }
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    entry[f"point{idx}"] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    }
                data.append(entry)

                index += 1

                with open(json_path, "w") as file:
                    json.dump(data, file, indent=4)

finally:
    pipeline.stop()
    hands.close()
    cv2.destroyAllWindows()
