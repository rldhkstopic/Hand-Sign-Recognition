import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0) # 캠 번호 확인하고 입력하세요.

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        color_image = cv2.flip(frame, 1) # 이미지 반전 : 필요없으면 삭제

        results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Webcam Feed', color_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key == 13:
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
    cap.release()
    hands.close()
    cv2.destroyAllWindows()
