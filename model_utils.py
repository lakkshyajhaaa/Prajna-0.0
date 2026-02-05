import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load trained model ----------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

class_names = ["day", "night"]

# ---------- Image preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- Core functions ----------
def get_probabilities(image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()[0]

def confidence(probs):
    return float(np.max(probs))

def margin(probs):
    p = np.sort(probs)[::-1]
    return float(p[0] - p[1])

def certainty_from_entropy(probs):
    eps = 1e-9
    probs = np.clip(probs, eps, 1)
    entropy = -np.sum(probs * np.log(probs))
    return float(1 - entropy / np.log(len(probs)))

def responsibility_score(C, M, U):
    
    return 0.4 * C + 0.3 * M + 0.3 * U
