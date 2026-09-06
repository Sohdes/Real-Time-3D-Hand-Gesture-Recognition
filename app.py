import streamlit as st

#  Configuration
st.set_page_config(
    page_title="3D Hand Gesture Recognition",
    page_icon="🖐️",
    layout="centered"
)
# Application Header
# *******************

st.title(
    "🖐️ 3D Hand Gesture Recognition"
)
st.write(
    "Real-time hand gesture classification using MediaPipe + PyTorch"
)




# The Panel of Info
st.sidebar.title(
    "Project Information"
)

st.sidebar.write(
    """
    **Computer Vision Pipeline**

    - MediaPipe Hand Tracking
    - 3D Landmark Extraction
    - Feature Engineering
    - PyTorch Classification
    """
)

# The Panel of System
st.subheader(
    "System Status"
)


model_status_card, input_status_card, accuracy_status_card = st.columns(3)
with model_status_card:

    st.metric(
        "Model",
        "PyTorch"
    )

with input_status_card:
    st.metric(
        "Input",
        "3D Hand"
    )
with accuracy_status_card:

    st.metric(
        "Accuracy",
        "99%"
    )

# Camera
st.subheader(
    "Camera Feed"
)

st.info(
    "Camera integration will be added in the next step."


)