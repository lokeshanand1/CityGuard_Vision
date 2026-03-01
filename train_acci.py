import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# =========================
# DATASET PATHS
# =========================

train_dir = r"F:\traffic Project\trainiig\Traffic Accident_Video_Dataset\train\train"
val_dir = r"F:\traffic Project\trainiig\Traffic Accident_Video_Dataset\val\val"
test_dir = r"F:\traffic Project\trainiig\Traffic Accident_Video_Dataset\test\test"

# =========================
# CLASS LABELS
# =========================

classes = [
    "Backend",
    "Backend_rollover",
    "Frontend",
    "Frontend_rollover",
    "General_Augmented_Crash_Download",
    "Noaccident_norm_traffic",
    "sidehit",
    "sidehit_rollover"
]

class_to_idx = {cls: i for i, cls in enumerate(classes)}

# =========================
# LOAD YOLO MODEL
# =========================

print("Loading YOLO model...")
yolo = YOLO("yolov10m.pt")

# =========================
# DATASET CLASS
# =========================

class AccidentDataset(Dataset):

    def __init__(self, root_dir, seq_len=16):

        self.root_dir = root_dir
        self.seq_len = seq_len
        self.samples = []

        video_ext = (".mp4", ".avi", ".mov", ".mkv", ".MP4")

        for label in os.listdir(root_dir):

            label_path = os.path.join(root_dir, label)

            if label not in class_to_idx:
                continue

            for file in os.listdir(label_path):

                if file.endswith(video_ext):

                    video_path = os.path.join(label_path, file)

                    self.samples.append((video_path, class_to_idx[label]))

        print(f"Found {len(self.samples)} videos in {root_dir}")

    def extract_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)

        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        step = max(total_frames // self.seq_len, 1)

        for i in range(self.seq_len):

            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)

            ret, frame = cap.read()

            if ret:

                frame = cv2.resize(frame, (224,224))

                frames.append(frame)

        cap.release()

        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        video_path, label = self.samples[idx]

        frames = self.extract_frames(video_path)

        features = []

        for frame in frames:

            results = yolo(frame, verbose=False)[0]

            if len(results.boxes) > 0:

                box = results.boxes.xyxy.cpu().numpy()[0]

                features.append(box)

            else:

                features.append([0,0,0,0])

        # pad frames if missing
        while len(features) < self.seq_len:
            features.append([0,0,0,0])

        features = np.array(features)

        features = torch.from_numpy(features).float()

        return features, label


# =========================
# CREATE DATASETS
# =========================

train_dataset = AccidentDataset(train_dir)
val_dataset = AccidentDataset(val_dir)
test_dataset = AccidentDataset(test_dir)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

# =========================
# LSTM MODEL
# =========================

class AccidentLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(64, 8)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# =========================
# INITIALIZE MODEL
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AccidentLSTM().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# TRAINING
# =========================

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for features, labels in tqdm(train_loader):

        features = features.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader)}")


# =========================
# VALIDATION
# =========================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for features, labels in val_loader:

        features = features.to(device)

        labels = labels.to(device)

        outputs = model(features)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print("Validation Accuracy:", 100 * correct / total)

# =========================
# SAVE MODEL
# =========================

torch.save(model.state_dict(), "accident_detection_model.pth")

print("Model saved successfully!")