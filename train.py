import torch
from torch.utils.data import DataLoader
from dataset import TBDataset
from model import MultiModalTBModel
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = TBDataset("data/clinical.csv", "data/images")
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0  
)

print(f"Total samples: {len(dataset)}")
print(f"Total batches per epoch: {len(loader)}")

model = MultiModalTBModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 2

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, clinical, labels) in enumerate(loader):
        images = images.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, clinical)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 25 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"Batch [{batch_idx}/{len(loader)}] | "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = running_loss / len(loader)
    print(f" Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}\n")

torch.save(model.state_dict(), "tb_multimodal_model.pth")
print("🎉 Training complete. Model saved as tb_multimodal_model.pth")