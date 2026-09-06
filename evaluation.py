import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report




# Loading gesture feature dataset
# **************************
gesture_feature_table = pd.read_csv(
    "gesture_features_dataset.csv"
)


input_feature_matrix = gesture_feature_table.drop(
    "label",
    axis=1
).values


gesture_labels = gesture_feature_table[
    "label"
].values


# Create evaluation split
# **********************
_, evaluation_features, _, evaluation_labels = train_test_split(
    input_feature_matrix,
    gesture_labels,
    test_size=0.2,
    random_state=42
)
evaluation_features = torch.tensor(
    evaluation_features,
    dtype=torch.float32
)


# Gesture of Classification of Our Model
class GestureNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(65, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 3)
        )
    def forward(self, feature_vector):

        return self.network(feature_vector)

# Load trained weights
# ********************

trained_gesture_model = GestureNet()

trained_gesture_model.load_state_dict(
    torch.load("gesture_model.pth")
)
trained_gesture_model.eval()

# Run prediction
# ***********************

with torch.no_grad():

    model_raw_scores = trained_gesture_model(
        evaluation_features
    )


    predicted_gesture_ids = torch.argmax(
        model_raw_scores,
        dim=1
    )



predicted_gesture_ids = predicted_gesture_ids.numpy()


# Evaluation

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        evaluation_labels,
        predicted_gesture_ids
    )
)

print("\nClassification Report:")
print(
    classification_report(
        evaluation_labels,
        predicted_gesture_ids,
        target_names=[
            "Open Hand",
            "Fist",
            "Thumbs Up"
        ]
    )
)