# Real-Time 3D Hand Gesture Recognition

A real-time hand gesture recognition system using MediaPipe Hands, 
PyTorch, OpenCV, and Streamlit.

## Overview

This project implements a vision-based human-computer interaction (HCI)
system that recognizes hand gestures from webcam input.

The system uses MediaPipe Hands to extract 3D hand landmarks from camera
frames. The extracted landmark coordinates are converted into numerical
feature vectors and classified using a lightweight PyTorch neural network.

The final application provides real-time gesture recognition through a
Streamlit interface.

## Features

- Real-time hand detection from webcam input
- 3D hand landmark extraction using MediaPipe Hands
- Feature extraction from hand geometry
- Neural network-based gesture classification
- Confidence score estimation
- Real-time visualization using Streamlit

## Supported Gestures

The system recognizes three predefined gestures:

- Open Hand → START
- Fist → STOP
- Thumbs Up → SELECT

## System Pipeline

The complete processing pipeline consists of:

1. Capturing video frames from the webcam
2. Detecting hand landmarks using MediaPipe Hands
3. Extracting numerical features from landmarks
4. Classifying gestures using the PyTorch neural network
5. Displaying results through the Streamlit interface

## Model Architecture

The gesture classification model is a lightweight fully connected neural
network.


65 → 128 → 64 → 3


Input:
- 65-dimensional feature vector

Output:
- Open Hand
- Fist
- Thumbs Up

## Results

The trained model achieved approximately:

**93% accuracy**

The model was evaluated using classification metrics including confusion
matrix and classification report.

## Technologies

- Python
- PyTorch
- MediaPipe Hands
- OpenCV
- Streamlit
- NumPy
- Scikit-learn
- LaTeX

## Project Structure


Real-Time-3D-Hand-Gesture-Recognition/

├── train.py
├── evaluation.py
├── feature_extraction.py
├── gesture_features.py
├── streamlit_gesture_app.py
├── gesture_model.pth
├── requirements.txt
├── Report.pdf
└── README.md


## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
Running the Application

Run the Streamlit interface:

streamlit run streamlit_gesture_app.py
Report

The complete project report is available in:

Report.pdf
Author

Soheil Hemmat
