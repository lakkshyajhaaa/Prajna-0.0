# प्रज्ञा (Prajñā)

**प्रज्ञा** is a responsibility-aware AI system that evaluates not only what a machine learning model predicts, but whether that prediction should be trusted before action.

Instead of relying purely on accuracy, the system introduces a *responsibility layer* that measures confidence, uncertainty, and decisiveness at the level of individual predictions.

---

## Motivation

Most AI systems are forced to produce an output even when they are uncertain.  
In real-world, safety-critical settings, a *wrong but confident* prediction can be more harmful than no prediction at all.

**प्रज्ञा** is built on the idea that an AI system should be able to:
- act when it is confident,
- request human review when uncertain,
- and refuse to decide when reliability is low.

This aligns with the principles of Responsible AI and human-in-the-loop decision making.

---

## What the system does

The system performs image classification (demonstrated using a day/night task) and computes a responsibility score for each prediction based on:

- **Confidence** – how strongly the model favors its top prediction  
- **Margin** – how clearly one class dominates over others  
- **Entropy** – how uncertain the prediction distribution is  
- **Certainty** – normalized inverse of entropy  

These metrics are combined into a single **responsibility score**, which drives a three-level decision:

- **ACCEPT** – prediction is trusted  
- **REVIEW** – prediction requires human verification  
- **REJECT** – system abstains from deciding  

---

## Key features

- Responsibility-aware decision layer over model predictions  
- Explainable, per-sample trust metrics  
- Identification of high-risk errors (wrong but accepted cases)  
- Interactive Streamlit interface for analysis and demonstration  
- Dataset-wide evaluation and single-image inspection  

---

## Technology stack

- Python  
- PyTorch  
- Torchvision (ResNet18)  
- Streamlit  
- NumPy, Pandas  

---

## Running the application

```bash
pip install -r requirements.txt
streamlit run app.py

## Dataset

The demonstration task uses a publicly available day–night image dataset sourced from Kaggle.

The dataset is not included in this repository to avoid redistribution and storage concerns.  
Interested users can obtain it directly from Kaggle:

- https://www.kaggle.com/datasets/ibrahimalobaid/day-and-night-image
