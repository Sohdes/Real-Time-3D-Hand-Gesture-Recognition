import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import math
import time


# App Configuration
# ***************
st.set_page_config(
    page_title="3D Hand Gesture Recognition",
    page_icon="🖐️"
)

st.title("🖐️ Real-Time 3D Hand Gesture Recognition")

st.write(
    "Hand gesture recognition using MediaPipe and also PyTorch"
)

# MediaPipe Setup
# Initialize MediaPipe hand detector.
# It extracts 21 hand landmarks from the camera frame.
# Each landmark contains x, y, z coordinates.
# ***************
hands_module = mp.solutions.hands
@st.cache_resource
def create_hand_detector():

    return hands_module.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    )

hand_detector = create_hand_detector()
drawing_utils = mp.solutions.drawing_utils

# Neural Network
# Simple fully connected neural network.
# Input: 65 extracted hand features
# Output: 3 gesture classes
# ***************
class GestureClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(65, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, input_data):
        return self.network(input_data)



@st.cache_resource
def load_gesture_model():

    model = GestureClassifier()

    model.load_state_dict(
        torch.load(
            "gesture_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model



gesture_model = load_gesture_model()

# Gesture Mapping
# ***************
gesture_names = {
    0: "Open Hand",
    1: "Fist",
    2: "Thumbs Up"
}


gesture_commands = {
    "Open Hand": "START",
    "Fist": "STOP",
    "Thumbs Up": "SELECT"
}

# Feature Extraction
# Convert MediaPipe landmarks into numerical features.
# 21 landmarks × 3 coordinates = 63 features.
# Two extra distance features are added for thumb information.
# ***************
def calculate_distance(point_a, point_b):

    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2 +
        (point_a[2] - point_b[2]) ** 2
    )


def extract_features(hand_landmarks):

    landmark_data = []

    for point in hand_landmarks.landmark:

        landmark_data.extend([
            point.x,
            point.y,
            point.z
        ])
    points_3d = []

    for index in range(0, 63, 3):

        points_3d.append([
            landmark_data[index],
            landmark_data[index + 1],
            landmark_data[index + 2]
        ])


    thumb_tip = points_3d[4]
    thumb_joint = points_3d[3]
    thumb_base = points_3d[2]


    landmark_data.append(
        calculate_distance(
            thumb_tip,
            thumb_base
        )
    )


    landmark_data.append(
        calculate_distance(
            thumb_tip,
            thumb_joint
        )
    )


    return landmark_data


# Streamlit Interface
# ***************
st.subheader("Camera Feed")
camera_window = st.empty()
status_window = st.empty()
gesture_window = st.empty()
confidence_window = st.empty()
action_window = st.empty()
fps_window = st.empty()
st.subheader("System Information")

st.info(
"""
Model: PyTorch Neural Network

Architecture:
65 → 128 → 64 → 3

Input:
3D Hand Landmarks + Distance Features

Vision:
MediaPipe Hands

Classes:
Open Hand / Fist / Thumbs Up

Accuracy:
93%
"""
)
start_camera = st.button("▶ Start Camera")
stop_camera = st.button("⏹ Stop Camera")

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "camera" not in st.session_state:
    st.session_state.camera = None

if start_camera:
    st.session_state.camera_running = True
    st.session_state.camera = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

if stop_camera:
    st.session_state.camera_running = False

    if st.session_state.camera is not None:
        st.session_state.camera.release()

    st.session_state.camera = None


# Camera Processing
# ***************

last_gesture = ""
last_action = ""
last_confidence = 0
if st.session_state.camera_running:
    camera = st.session_state.camera
    if camera is None or not camera.isOpened():
        st.error("Camera could not be opened")
        st.stop()
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )
    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )
    last_time = time.time()

    while st.session_state.camera_running:
        success, frame = camera.read()
        if not success:
            st.error("Camera connection failed")
            break
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
        result = hand_detector.process(
            rgb_frame
        )
        current_status = "🟡 Waiting for hand..."

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                current_status = "🟢 Hand detected"
                drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    hands_module.HAND_CONNECTIONS
                )
                feature_values = extract_features(hand_landmarks)
                input_tensor = torch.tensor(
                    feature_values,
                    dtype=torch.float32
                )
                input_tensor = input_tensor.unsqueeze(0)
                with torch.no_grad():
                    prediction_output = gesture_model(input_tensor)
                    prediction_probability = torch.softmax(
                        prediction_output,
                        dim=1
                    )
                    confidence,class_id = torch.max(
                        prediction_probability,
                        dim=1
                    )
                new_gesture = gesture_names[class_id.item()]
                new_confidence = confidence.item()*100
                if new_confidence > last_confidence-10:
                    last_gesture = new_gesture
                    last_confidence = new_confidence
                detected_gesture = last_gesture
                confidence_score = last_confidence
                detected_action = gesture_commands[detected_gesture]
                if confidence_score >= 90:
                    confidence_status = "🟢"
                elif confidence_score >= 70:
                    confidence_status = "🟡"
                else:
                    confidence_status = "🔴"
                cv2.putText(
                    frame,
                    detected_gesture,
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2
                )
                cv2.putText(
                    frame,
                    f"Action: {detected_action}",
                    (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )
                gesture_window.markdown(
                    f"## Gesture: {detected_gesture}"
                )
                confidence_window.markdown(
                    f"## Confidence: {confidence_status} {confidence_score:.1f}%"
                )
                action_window.markdown(
                    f"## Action: {detected_action}"
                )
        else:
            gesture_window.markdown("## Gesture: -")
            confidence_window.markdown("## Confidence: -")
            action_window.markdown("## Action: -")
        status_window.markdown(
            f"## Status: {current_status}"
        )
        current_time = time.time()
        fps_value = 1/(current_time-last_time)
        last_time = current_time
        fps_window.markdown(
            f"## FPS: {fps_value:.1f}"
        )
        camera_window.image(
            frame,
            channels="BGR"
        )
        if cv2.waitKey(1)&0xFF == ord("q"):
            st.session_state.camera_running = False
            break
    camera.release()