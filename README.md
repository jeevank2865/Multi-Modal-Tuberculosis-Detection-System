# 🫁 Multi-Modal Tuberculosis Detection System

## Project Description
The **Multi-Modal Tuberculosis Detection System** is an AI-powered medical screening application developed using **Deep Learning and Streamlit**.  
The system detects **Tuberculosis (TB)** by analyzing **Chest X-ray images** along with **clinical attributes** such as age, fever, cough, and weight loss.

The project follows a **multi-modal deep learning approach**, where:
- A **Convolutional Neural Network (CNN)** extracts visual features from chest X-rays.
- A **fully connected neural network** processes clinical features.
- Both feature sets are fused to produce a final diagnosis.

This project is intended for **academic and educational purposes** and demonstrates how combining medical images with clinical data can improve diagnostic accuracy.

> Dataset Source: Kaggle – Tuberculosis Chest X-ray Dataset

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Theory](#Theory)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Deployment](#Deployment)
- [Contributing](#Contributions)



---

## Overview
The system performs **binary classification**:
- **Tuberculosis (TB)**
- **Normal (Healthy)**

The trained deep learning model is deployed using a **Streamlit web application**, allowing users to:
- Upload a chest X-ray image
- Optionally enter clinical details
- Receive a diagnosis with a confidence score

---

## Features
- Multi-modal deep learning (Image + Clinical data)
- CNN-based Chest X-ray feature extraction
- Clinical data fusion for enhanced prediction
- Streamlit-based interactive web interface
- Confidence score for predictions
- Accepts only valid chest X-ray images
- Clean, professional, and demo-ready UI
- Deployable on Streamlit Cloud

---

## Theory

### What is Tuberculosis?
Tuberculosis (TB) is a contagious bacterial infection that primarily affects the lungs. Chest X-ray imaging is one of the most widely used diagnostic tools for early TB screening.

---

### What is Multi-Modal Deep Learning?
Multi-modal deep learning combines different types of data sources to improve prediction accuracy.

In this project:
- **Image Modality:** Chest X-ray images
- **Clinical Modality:** Age, fever, cough, weight loss

Both modalities are processed separately and fused before classification.

---

### Why Use Deep Learning for TB Detection?
- Learns complex radiological patterns automatically
- Reduces human interpretation errors
- Enables scalable screening solutions
- Suitable for web-based deployment

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jeevank2865/Multi-Modal-Tuberculosis-Detection-System.git
    cd Multi-Modal-Tuberculosis-Detection-System
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3. Install required dependencies:
   ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
   ```bash
    streamlit run app.py
    ```
## Usage

1. Launch the application:
    ```bash
    streamlit run app.py
    ```
2. Upload a Chest X-ray image
   
3. (Optional) Enter clinical details:
	-	Age
	-	Fever
	-	Cough
	-	Weight loss
 	
4. Click Analyze Chest X-ray
  	
5. View:
	-	Diagnosis (TB / Normal)
	-	Confidence score

## Dataset

The project uses a publicly available Tuberculosis Chest X-ray Dataset from Kaggle.

Dataset structure:
	- TB images
	-	Normal images

Clinical data is synthetically generated for academic demonstration purposes.

Dataset source: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

## Model 

	-	CNN Backbone: ResNet-based feature extractor
	-	Clinical Network: Fully connected layers
	-	Fusion Layer: Concatenates image & clinical features
	-	Output: Binary classification (TB / Normal)

Loss Function:
	-	Cross Entropy Loss

Optimizer:
	-	Adam Optimizer

Framework:
	-	PyTorch
  
## Results

The trained model achieves high classification accuracy on the dataset.

Live Demo: https://multi-modal-tuberculosis-detection-system-vyjaztxp4wvsekcapp83.streamlit.app/
<br>
<br>
<br>
<img width="1470" height="835" alt="Screenshot 2026-02-18 at 3 36 02 PM" src="https://github.com/user-attachments/assets/4c29b70f-f291-40c0-af5e-2077fdb22a60" />

<br>
<br>
<br>
<br>

## Deployment

The project is deployed using Streamlit Cloud.

 Deployment steps:

	- Push code and model to GitHub
	- Configure requirements.txt
	- Deploy via Streamlit Cloud dashboard
	- Access via public URLContributing

## Contributions

Possible future improvements:
	- Use real clinical datasets
	- Add severity classification
	- Improve explainability (Grad-CAM)
	- Integrate doctor feedback system
	- Add user authentication
