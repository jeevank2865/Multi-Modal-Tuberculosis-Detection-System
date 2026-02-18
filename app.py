import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import MultiModalTBModel

st.set_page_config(
    page_title="Tuberculosis Detection System",
    page_icon="🫁",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align:center;'>🫁 Tuberculosis Detection System</h1>
    <p style='text-align:center; color:grey;'>
    Multi-Modal Deep Learning using Chest X-ray and Clinical Data
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = MultiModalTBModel().to(device)
    model.load_state_dict(
        torch.load("tb_multimodal_model.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def is_xray_image(pil_image):
    img = np.array(pil_image)
    if img.ndim == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (
            np.mean(np.abs(r - g)) > 15 or
            np.mean(np.abs(r - b)) > 15 or
            np.mean(np.abs(g - b)) > 15
        ):
            return False
    if img.std() < 20:
        return False
    return True

st.sidebar.header("🧑 Patient Information")

use_clinical = st.sidebar.checkbox("Include clinical details (optional)", value=False)

if use_clinical:
    age = st.sidebar.slider("Age", 1, 100, 30)
    fever = st.sidebar.selectbox("Fever", ["No", "Yes"])
    cough = st.sidebar.selectbox("Cough", ["No", "Yes"])
    weight_loss = st.sidebar.selectbox("Weight Loss", ["No", "Yes"])

    fever = 1 if fever == "Yes" else 0
    cough = 1 if cough == "Yes" else 0
    weight_loss = 1 if weight_loss == "Yes" else 0
else:
    age = 30
    fever = 0
    cough = 0
    weight_loss = 0

st.sidebar.markdown("---")
st.sidebar.info(
    "Clinical details are optional.\n\n"
    "If not provided, image-based analysis is used."
)

st.subheader("📤 Upload Chest X-ray Image")

uploaded_file = st.file_uploader(
    "Supported formats: PNG, JPG, JPEG",
    type=["png", "jpg", "jpeg"]
)

st.markdown("---")

if st.button("🔍 Analyze Chest X-ray", use_container_width=True):

    if uploaded_file is None:
        st.warning("Please upload a chest X-ray image.")
    else:
        image = Image.open(uploaded_file).convert("RGB")

        if not is_xray_image(image):
            st.error(
                "❌ Invalid image.\n\n"
                "Please upload a valid chest X-ray."
            )
        else:
            st.image(
                image,
                caption="Uploaded Chest X-ray",
                use_container_width=True
            )

            image_tensor = transform(image).unsqueeze(0).to(device)
            clinical_tensor = torch.tensor(
                [[age, fever, cough, weight_loss]],
                dtype=torch.float32
            ).to(device)

            with torch.no_grad():
                output = model(image_tensor, clinical_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(probs, dim=1)

            confidence = confidence.item()
            prediction = prediction.item()

            st.markdown("### 🧪 Diagnostic Result")

            if confidence < 0.75:
                st.warning(
                    f"⚠️ Inconclusive result (Confidence: {confidence:.2f}). "
                    "Please upload a clearer X-ray."
                )
            elif prediction == 1:
                st.error(
                    f"🩺 Tuberculosis Detected\n\n"
                    f"Confidence: {min(confidence, 0.99):.2f}"
                )
            else:
                st.success(
                    f"🩺 No Tuberculosis Detected\n\n"
                    f"Confidence: {confidence:.2f}"
                )

st.markdown("---")
st.caption(
    "⚠️ This application is for educational purposes only."
)