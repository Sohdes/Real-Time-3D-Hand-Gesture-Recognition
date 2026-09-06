import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# Load dataset
#****************************
data = pd.read_csv("gesture_features_dataset.csv")
# Spliting features and labels
X = data.drop("label", axis=1).values
y = data["label"].values


# Train & Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42)

# Convering to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)


X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# Loader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)


# Our Nural Network
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


    def forward(self, x):
        return self.network(x)

model = GestureNet()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}"
    )


# Test accuracy
with torch.no_grad():

    predictions = model(X_test)

    predicted_labels = torch.argmax(
        predictions,
        dim=1
    )
    accuracy = (
        predicted_labels == y_test
    ).float().mean()

print(
    "Accuracy:",
    accuracy.item()
)

# Saving the model
torch.save(
    model.state_dict(),
    "gesture_model.pth"
)

print("Model saved!")