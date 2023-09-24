import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def detect_hand_landmarks(image):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        else:
            return None

def draw_landmarks(image, landmarks):
    if landmarks:
        for i, landmark in enumerate(landmarks.landmark):
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)

def print_landmarks(landmarks):
    if landmarks:
        for i, landmark in enumerate(landmarks.landmark):
            print(f"Landmark {i}: ({landmark.x}, {landmark.y}, {landmark.z})")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = detect_hand_landmarks(frame)
    draw_landmarks(frame, landmarks)
    cv2.imshow("Hand Landmarks", frame)

    print_landmarks(landmarks)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()