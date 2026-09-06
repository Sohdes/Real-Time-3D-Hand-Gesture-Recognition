import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import math


def calculate_distance(point_a, point_b):

    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2 +
        (point_a[2] - point_b[2]) ** 2
    )



def extract_realtime_features(landmarks):

    features = []

    # original 63 values
    features.extend(landmarks)


    # convert to 21 points

    points = []

    for i in range(0, 63, 3):

        points.append([
            landmarks[i],
            landmarks[i+1],
            landmarks[i+2]
        ])


    # thumb features

    thumb_tip = points[4]
    thumb_joint = points[3]
    thumb_base = points[2]


    thumb_tip_to_base = calculate_distance(
        thumb_tip,
        thumb_base
    )


    thumb_tip_to_joint = calculate_distance(
        thumb_tip,
        thumb_joint
    )


    features.append(
        thumb_tip_to_base
    )

    features.append(
        thumb_tip_to_joint
    )


    return features


# PyTorch Gesture Classification Model
# ************************************

class HandGestureClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(65, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 3)
        )

    def forward(self, landmark_features):
        return self.network(landmark_features)

# Loading ourr trained model

gesture_classifier = HandGestureClassifier()

gesture_classifier.load_state_dict(
    torch.load("gesture_model.pth")
)

gesture_classifier.eval()

# Names of Gesture

gesture_names = {
    0: "Open Hand",
    1: "Fist",
    2: "Thumbs Up"
}


# MediaPipe Hand Tracking Setup


mp_hand_module = mp.solutions.hands
landmark_drawer = mp.solutions.drawing_utils

hand_tracker = mp_hand_module.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)


# Camera Setup
# ************************************

webcam_stream = cv2.VideoCapture(0)


# Real-Time Prediction Loop
# ************************************

while True:
    camera_available, camera_frame = webcam_stream.read()
    if not camera_available:
        break

    rgb_camera_frame = cv2.cvtColor(
        camera_frame,
        cv2.COLOR_BGR2RGB
    )

    hand_detection_result = hand_tracker.process(
        rgb_camera_frame
    )


    if hand_detection_result.multi_hand_landmarks:
        for detected_hand in hand_detection_result.multi_hand_landmarks:
            # Draw hand skeleton
            landmark_drawer.draw_landmarks(
                camera_frame,
                detected_hand,
                mp_hand_module.HAND_CONNECTIONS
            )



            # Extract 3D landmark features
            landmark_vector = []

            for landmark_point in detected_hand.landmark:
                landmark_vector.extend([
                    landmark_point.x,
                    landmark_point.y,
                    landmark_point.z
                ])

            model_features = extract_realtime_features(
                landmark_vector
            )



            # Converting features to PyTorch tensor
            model_input = torch.tensor(
                model_features,
                dtype=torch.float32
            )
            model_input = model_input.unsqueeze(0)


            # Our Model Prediction

            with torch.no_grad():

                prediction_scores = gesture_classifier(
                    model_input
                )
                probabilities = torch.softmax(
                    prediction_scores,
                    dim=1
                )

                confidence, predicted_class = torch.max(
                    probabilities,
                    dim=1
                )

                confidence = confidence.item()

                predicted_class = predicted_class.item()

            detected_gesture = (
                    gesture_names[predicted_class]
                    + f" {confidence * 100:.1f}%"
            )



            # Displaying
            cv2.putText(
                camera_frame,
                f"Gesture: {detected_gesture}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


    cv2.imshow(
        "3D Hand Gesture Recognition",
        camera_frame
    )


    # Exit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


webcam_stream.release()
cv2.destroyAllWindows()