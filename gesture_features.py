import pandas as pd
import math


def calculate_distance(point_a, point_b):
    """
    Calculate 3D distance between two landmarks
    """

    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2 +
        (point_a[2] - point_b[2]) ** 2
    )



def extract_hand_features(landmarks):

    features = []


    # Original 63 landmark values
    features.extend(landmarks)


    # Convert flat list to 21 xyz points

    points = []

    for i in range(0, 63, 3):

        points.append([
            landmarks[i],
            landmarks[i+1],
            landmarks[i+2]
        ])


    # ----------------------------
    # Thumb features
    # ----------------------------

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



dataset = pd.read_csv("gesture_data.csv")

new_rows = []

for _, row in dataset.iterrows():

    label = row["label"]

    landmarks = row.drop("label").values.astype(float)

    extracted_features = extract_hand_features(
        landmarks
    )

    new_rows.append(
        [label] + extracted_features
    )


columns = ["label"]

for i in range(63):
    columns.append(f"feature_{i}")


columns.extend([
    "thumb_tip_base_distance",
    "thumb_tip_joint_distance"
])


new_dataset = pd.DataFrame(
    new_rows,
    columns=columns
)


new_dataset.to_csv(
    "gesture_features_dataset.csv",
    index=False
)


print("Feature dataset created!")
print(new_dataset.shape)