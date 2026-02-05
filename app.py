# ==========================================
# प्रज्ञा — RESPONSIBLE AI DECISION SYSTEM
# ==========================================

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import streamlit as st
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights


# -------------------------
# STREAMLIT CONFIG
# -------------------------
st.set_page_config(
    page_title="प्रज्ञा",
    layout="wide"
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# LOAD TRAINED MODEL
# -------------------------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

class_names = ["day", "night"]


# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# LOAD TEST DATA
# -------------------------
test_data = datasets.ImageFolder("dataset/test", transform=transform)


# -------------------------
# RESPONSIBILITY METRICS
# -------------------------
def get_probabilities(image):
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()[0]

def confidence(probs):
    return float(np.max(probs))

def margin(probs):
    p = np.sort(probs)[::-1]
    return float(p[0] - p[1])

def entropy_value(probs):
    eps = 1e-9
    probs = np.clip(probs, eps, 1)
    return float(-np.sum(probs * np.log(probs)))

def certainty_from_entropy(probs):
    H = entropy_value(probs)
    return float(1 - H / np.log(len(probs)))

def responsibility_score(C, M, U):
    return 0.4 * C + 0.3 * M + 0.3 * U

def decision(R):
    if R >= 0.8:
        return "ACCEPT"
    elif R >= 0.5:
        return "REVIEW"
    else:
        return "REJECT"


# -------------------------
# UI HEADER (UPGRADED)
# -------------------------
st.markdown("""
<h1 style='font-size:48px;'>प्रज्ञा <span style="font-size:22px; color:gray;">0.0</span></h1>
<p style="font-size:20px;">Prajñā — Responsible AI Decision Framework</p>
<p style="color:#9ca3af; max-width:700px;">
An explainable system that evaluates not only <b>what</b> the model predicts,
but <b>whether that prediction should be trusted</b>.
</p>
""", unsafe_allow_html=True)


# -------------------------
# MODE SELECTION
# -------------------------
mode = st.radio(
    "Select Mode",
    ["Analyze Test Dataset", "Check Uploaded Image"],
    horizontal=True
)


# =====================================================
# MODE 1: DATASET ANALYSIS
# =====================================================
if mode == "Analyze Test Dataset":

    if st.button("Run Responsibility Analysis on Test Set"):

        responsibility_scores = []
        decisions = []
        predicted_classes = []
        true_classes = []
        image_paths = []

        for idx in range(len(test_data)):
            image, true_label = test_data[idx]
            image = image.to(device)

            probs = get_probabilities(image)
            pred_idx = np.argmax(probs)

            C = confidence(probs)
            M = margin(probs)
            U = certainty_from_entropy(probs)
            R = responsibility_score(C, M, U)

            responsibility_scores.append(R)
            decisions.append(decision(R))
            predicted_classes.append(class_names[pred_idx])
            true_classes.append(class_names[true_label])
            image_paths.append(test_data.samples[idx][0])

        results_df = pd.DataFrame({
            "Image": image_paths,
            "True Class": true_classes,
            "Predicted Class": predicted_classes,
            "Responsibility Score": responsibility_scores,
            "Decision": decisions
        })

        st.success("Analysis complete")

        st.subheader("Responsibility Distribution")
        st.bar_chart(results_df["Responsibility Score"])

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Decision Breakdown")
            st.bar_chart(results_df["Decision"].value_counts())
        with col2:
            st.subheader("Prediction Distribution")
            st.bar_chart(results_df["Predicted Class"].value_counts())

        st.subheader("Detailed Results")
        st.dataframe(results_df)

        wrong_but_accepted = results_df[
            (results_df["True Class"] != results_df["Predicted Class"]) &
            (results_df["Decision"] == "ACCEPT")
        ]

        st.subheader("High-Risk Errors (Wrong but Accepted)")
        if len(wrong_but_accepted) == 0:
            st.success("No high-risk cases found")
        else:
            st.error(f"{len(wrong_but_accepted)} high-risk cases detected")
            st.dataframe(wrong_but_accepted)

        st.subheader("Summary")
        st.write(f"Total Images: {len(results_df)}")
        st.write(f"Accepted: {(results_df['Decision'] == 'ACCEPT').sum()}")
        st.write(f"Reviewed: {(results_df['Decision'] == 'REVIEW').sum()}")
        st.write(f"Rejected: {(results_df['Decision'] == 'REJECT').sum()}")


# =====================================================
# MODE 2: UPLOADED IMAGE
# =====================================================
if mode == "Check Uploaded Image":

    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        # Resize image for UI display only
        display_image = image.copy()
        display_image.thumbnail((600, 600))  # max width & height

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(display_image, caption="Uploaded Image")

            
        image_tensor = transform(image).to(device)
        probs = get_probabilities(image_tensor)

        pred_idx = np.argmax(probs)
        predicted = class_names[pred_idx]

        C = confidence(probs)
        M = margin(probs)
        H = entropy_value(probs)
        U = certainty_from_entropy(probs)
        R = responsibility_score(C, M, U)
        decision_label = decision(R)

        # Prediction
        st.subheader("Model Prediction")
        st.write(f"**Predicted Class:** {predicted.upper()}")

        st.dataframe(pd.DataFrame({
            "Class": class_names,
            "Probability": np.round(probs, 4)
        }), use_container_width=True)

        # Metrics
        st.subheader("Responsibility Metrics")
        st.dataframe(pd.DataFrame({
            "Metric": ["Confidence", "Margin", "Entropy", "Certainty", "Responsibility"],
            "Value": [C, M, H, U, R],
            "Formula": ["max(p)", "p₁ − p₂", "−Σp log(p)", "1 − H/log(K)", "0.4C+0.3M+0.3U"]
        }).round(3), use_container_width=True)

        # Decision
        st.subheader("Final Decision")
        if decision_label == "ACCEPT":
            st.success("ACCEPT — prediction is trusted")
        elif decision_label == "REVIEW":
            st.warning("REVIEW — human verification recommended")
        else:
            st.error("REJECT — prediction not trusted")
