import cv2
import numpy as np
import mediapipe as mp
import torch
from PIL import ImageFont, ImageDraw, Image
from hangul_utils import join_jamos

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# font_path = './ttf_files/BMDOHYEON_ttf.ttf'
# font_path = './ttf_files/BMJUA_ttf.ttf'
# font_path = './ttf_files/NanumGothic.ttf'

font_path = './ttf_files/휴먼매직체.ttf'
font = ImageFont.truetype(font_path, 30)










# Normalization
def normalize_landmarks(landmarks):
    landmarks_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks], dtype=np.float32)
    norm_landmarks = landmarks_array - np.mean(landmarks_array, axis=0)
    max_dist = np.max(np.linalg.norm(norm_landmarks, axis=1))
    norm_landmarks = norm_landmarks / max_dist
    return norm_landmarks



# 왼손인지 오른손인지 할당
def assign_hands_by_position(multi_hand_landmarks, image_width):
    # Assuming that the wrist is the first landmark in the hand landmarks list
    wrist_landmark_index = 0

    hands_x_positions = [
        hand.landmark[wrist_landmark_index].x for hand in multi_hand_landmarks
    ]

    left_hand_index = np.argmin(hands_x_positions)
    right_hand_index = 1 - left_hand_index  # The other hand is the right hand


    if hands_x_positions[left_hand_index] * image_width < hands_x_positions[right_hand_index] * image_width:
        hand_landmarks_0 = multi_hand_landmarks[left_hand_index]
        hand_landmarks_1 = multi_hand_landmarks[right_hand_index]
    else:
        hand_landmarks_0 = multi_hand_landmarks[right_hand_index]
        hand_landmarks_1 = multi_hand_landmarks[left_hand_index]

    return hand_landmarks_0, hand_landmarks_1






# 현재 손가락과 엄지가 닿았는지 확인하기위한 함수 calculate_distance, check_fingers_touching
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1[0] - landmark2[0])**2 + (landmark1[1] - landmark2[1])**2 + (landmark1[2] - landmark2[2])**2)

def check_fingers_touching(thumb_tip,index_finger_tip,middle_finger_tip,):
    THUMB_INDEX_THRESHOLD = 0.22
    THUMB_MIDDLE_THRESHOLD = 0.22

    thumb_index_distance = calculate_distance(thumb_tip, index_finger_tip)
    thumb_middle_distance = calculate_distance(thumb_tip, middle_finger_tip)

    thumb_index_touching = thumb_index_distance < THUMB_INDEX_THRESHOLD
    thumb_middle_touching = thumb_middle_distance < THUMB_MIDDLE_THRESHOLD

    return thumb_index_touching, thumb_middle_touching


# 왼손의 시그널 : 원그리기
def draw_hand_circle(image, hand_landmarks, touching, color, thickness=2, radius=30,THUMB_TIP_IDX=4):

    if touching:
        thumb_tip = hand_landmarks.landmark[THUMB_TIP_IDX]
        
        # 이미지의 크기에 맞춰서 실제 픽셀 좌표로 변환
        image_width, image_height = image.shape[1], image.shape[0]
        thumb_tip_x = int(thumb_tip.x * image_width)
        thumb_tip_y = int(thumb_tip.y * image_height)
        
        # 원을 그립니다.
        cv2.circle(image, (thumb_tip_x, thumb_tip_y), radius, color, thickness)
        
        return image
    


# 쌍자음 처리
def make_double_consonant(consonant):
    if consonant == 'ㄱ':
        return 'ㄲ'
    elif consonant == 'ㄷ':
        return 'ㄸ'
    elif consonant == 'ㅂ':
        return 'ㅃ'
    elif consonant == 'ㅅ':
        return 'ㅆ'
    elif consonant == 'ㅈ':
        return 'ㅉ'
    else:
        return consonant

# 모음 처리
def vowel_compensation(recorded_letters):
    """
    ㅗㅐ = ㅙ
    ㅗㅏ = ㅘ
    ㅜㅔ = ㅞ
    ㅜㅓ = ㅝ
    """
    if recorded_letters[-2:] == 'ㅗㅐ':
        return recorded_letters[:-2] + 'ㅙ'
    elif recorded_letters[-2:] == 'ㅗㅏ':
        return recorded_letters[:-2] + 'ㅘ'
    elif recorded_letters[-2:] == 'ㅜㅔ':
        return recorded_letters[:-2] + 'ㅞ'
    elif recorded_letters[-2:] == 'ㅜㅓ':
        return recorded_letters[:-2] + 'ㅝ'
    else:
        return recorded_letters





# 오른쪽 흰화면에 글자출력
def draw_text(image, text, position, font=font, font_size=20, color=(0, 0, 0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # PIL 이미지에 텍스트 쓰기
    draw.text(position, text, font=font, fill=color)

    # PIL 이미지를 OpenCV 이미지로 다시 변환
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)



# 모델실행
def landmarks(data):
    return np.array([[landmark['x'], landmark['y'], landmark['z']] for landmark in data]).transpose()

def inference_model(norm_landmarks, model):
    hand_data = []
    for norm_data in norm_landmarks:
        tmp= {
            "x": float(norm_data[0]),
            "y": float(norm_data[1]),
            "z": float(norm_data[2])
        }
        hand_data.append(tmp)
    
    data = torch.tensor(landmarks(hand_data),dtype=torch.float32).unsqueeze(0).to(device)
    output = model(data).squeeze(0).detach().cpu().numpy()
    return output










