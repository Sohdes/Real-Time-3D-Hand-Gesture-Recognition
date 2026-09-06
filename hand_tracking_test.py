import cv2
import mediapipe as mp


# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Camera
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        break

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)

    # If hand detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Print 3D coordinates
            for id, lm in enumerate(hand_landmarks.landmark):
                print(
                    id,
                    "x:", round(lm.x, 3),
                    "y:", round(lm.y, 3),
                    "z:", round(lm.z, 3)
                )

    cv2.imshow("3D Hand Tracking Test", img)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()