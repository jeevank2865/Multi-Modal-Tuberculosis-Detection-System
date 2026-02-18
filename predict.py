import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import MultiModalTBModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = MultiModalTBModel().to(device)
model.load_state_dict(torch.load("tb_multimodal_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict(image_path, age, fever, cough, weight_loss):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    clinical = torch.tensor([[age, fever, cough, weight_loss]],
                            dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(image, clinical)
        prediction = torch.argmax(output, dim=1).item()

    return "TB" if prediction == 1 else "Normal"

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    age = int(input("Age: "))
    fever = int(input("Fever (1=yes, 0=no): "))
    cough = int(input("Cough (1=yes, 0=no): "))
    weight_loss = int(input("Weight loss (1=yes, 0=no): "))

    result = predict(image_path, age, fever, cough, weight_loss)
    print("\n🩺 Prediction Result:", result)
