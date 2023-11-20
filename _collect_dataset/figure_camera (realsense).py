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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

save_path = "dataset"
json_path = "dataset/hand_landmarks.json"

if not os.path.exists(save_path):
    os.makedirs(save_path)

Target = [ 
            {'key' : 'ㄱ', 'eng' : 'giyeok', 'label' : 0}, 
            {'key' : 'ㄴ', 'eng' : 'nieun', 'label' : 1}, 
            {'key' : 'ㄷ', 'eng' : 'digeut', 'label' : 2}, 
            {'key' : 'ㄹ', 'eng' : 'rieul', 'label' : 3}, 
            {'key' : 'ㅁ', 'eng' : 'mieum', 'label' : 4}, 
            {'key' : 'ㅂ', 'eng' : 'bieup', 'label' : 5}, 
            {'key' : 'ㅅ', 'eng' : 'siot', 'label' : 6}, 
            {'key' : 'ㅇ', 'eng' : 'ieung', 'label' : 7}, 
            {'key' : 'ㅈ', 'eng' : 'jieut', 'label' : 8}, 
            {'key' : 'ㅊ', 'eng' : 'chieut', 'label' : 9}, 
            {'key' : 'ㅋ', 'eng' : 'kieuk', 'label' : 10}, 
            {'key' : 'ㅌ', 'eng' : 'tieut', 'label' : 11}, 
            {'key' : 'ㅍ', 'eng' : 'pieup', 'label' : 12}, 
            {'key' : 'ㅎ', 'eng' : 'hieut', 'label' : 13}, 
            {'key' : 'ㅏ', 'eng' : 'a', 'label' : 14}, 
            {'key' : 'ㅑ', 'eng' : 'ya', 'label' : 15}, 
            {'key' : 'ㅓ', 'eng' : 'eo', 'label' : 16}, 
            {'key' : 'ㅕ', 'eng' : 'yeo', 'label' : 17}, 
            {'key' : 'ㅗ', 'eng' : 'o', 'label' : 18}, 
            {'key' : 'ㅛ', 'eng' : 'yo', 'label' : 19}, 
            {'key' : 'ㅜ', 'eng' : 'u', 'label' : 20}, 
            {'key' : 'ㅠ', 'eng' : 'yu', 'label' : 21}, 
            {'key' : 'ㅡ', 'eng' : 'eu', 'label' : 22}, 
            {'key' : 'ㅣ', 'eng' : 'i', 'label' : 23}, 
            {'key' : 'ㅔ', 'eng' : 'e', 'label' : 24}, 
            {'key' : 'ㅖ', 'eng' : 'ye', 'label' : 25}, 
            {'key' : 'ㅐ', 'eng' : 'ae', 'label' : 26}, 
            {'key' : 'ㅡ', 'eng' : 'yae', 'label' : 27}, 
            {'key' : 'ㅢ', 'eng' : 'ui', 'label' : 28}, 
            {'key' : 'ㅚ', 'eng' : 'oe', 'label' : 29}, 
            {'key' : 'ㅟ', 'eng' : 'wi', 'label' : 30}, 

        ]

n = 0
label = Target[n]['label']
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
                    "label": label,
                    "index": index
                }
                hand_data = []
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    tmp= {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    }
                    hand_data.append(tmp)
                
                entry["hand_data"] = hand_data
                data.append(entry)

                index += 1
                print(f"Saved {filename}")

                with open(json_path, "w") as file:
                    json.dump(data, file, indent=4)

        elif key == ord('c') or key == ord('C'):
            n += 1
            selected_char = Target[n]['eng']
            label = Target[n]['label']
            index = 1
            print(f"Selected character: {selected_char}")

finally:
    pipeline.stop()
    hands.close()
    cv2.destroyAllWindows()
