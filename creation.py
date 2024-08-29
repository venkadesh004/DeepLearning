import cv2
import mediapipe as mp
import json
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

count = 0

filename = 'data.json'
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ptr = 0
data = {}
def extract_features(landmarks):
    features = []
    for lm in landmarks:
        features.append(lm.x)
        features.append(lm.y)
    return features

letterList = []

while True and ptr < 3:
    print("Now for letter", alphabets[ptr], "Time:", count)
    success, img = cap.read()
    if not success:
        break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    result = hands.process(img_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_features(hand_landmarks.landmark)
            features = np.array(features).reshape(1, -1)
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(f'Landmark {id}: (x: {cx}, y: {cy})')

    cv2.imshow("Hand Gesture Detection", img)
    
    if cv2 .waitKey(10) & 0xFF == ord('s'):
        # cv2.imwrite(f'./images/B/frame_{count}.png', img)
        letterList.append(features.tolist())
        count += 1
        if (count == 5):
            data[alphabets[ptr]] = letterList
            letterList = []
            ptr += 1
            count = 0

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # print(data)
    
filename = 'data.json'
print(data)
with open(filename, 'w') as file:
    json.dump(data, file, indent=4)
    
print(f'Data saved to {filename}')

cap.release()
cv2.destroyAllWindows()
