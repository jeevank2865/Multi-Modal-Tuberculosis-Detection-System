import torch
from torch.utils.data import DataLoader
from dataset import TBDataset
from model import MultiModalTBModel
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
dataset = TBDataset("data/clinical.csv", "data/images")
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0   
)

model = MultiModalTBModel().to(device)
model.load_state_dict(torch.load("tb_multimodal_model.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, clinical, labels in loader:
        images = images.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        outputs = model(images, clinical)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["Normal", "TB"])

print("\n Evaluation Results")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)