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
- Overview
- Features
- Theory
- Installation
- Usage
- Dataset
- Model Architecture
- Results
- Deployment
- Contributing
- License

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
