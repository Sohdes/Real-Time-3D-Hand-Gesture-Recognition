import cv2
import mediapipe as mp
import csv
import os


# MediaPipe2

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# Save file
file_name = "gesture_data.csv"

if not os.path.exists(file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)

        header = ["label"]

        for i in range(21):
            header += [
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ]

        writer.writerow(header)


# Choose gesture
print("Choose gesture:")
print("0 = Open Hand")
print("1 = Fist")
print("2 = Thumbs Up")

label = int(input("Enter label: "))


cap = cv2.VideoCapture(0)

print("Press S to save sample")
print("Press Q to quit")


while True:

    success, frame = cap.read()

    if not success:
        break


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)


    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )


            key = cv2.waitKey(1)


            if key == ord("s"):

                data = [label]

                for lm in hand_landmarks.landmark:
                    data.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])


                with open(file_name, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)


                print("Sample saved")


    cv2.imshow(
        "Collect Gesture Data",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()