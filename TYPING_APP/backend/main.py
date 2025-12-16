from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from preprocessing import pre_traitement, prepare_data_emotion_user_sequence, standardize_data
from model import ClassificationKeystrokeModel
import numpy as np
import random
from fastapi import FastAPI

# --------------------
# Fixer la seed (comme à l'entraînement)
# --------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # si plusieurs GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # à mettre **avant tout traitement**

# --------------------
# Global app + config
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://localhost"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Load model once at startup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClassificationKeystrokeModel(
    num_layer=1,
    d_model=6,
    k=32,
    heads=3,
    _heads=5,
    seq_len=65,
    num_classes=4,
    inner_dropout=0.2
)

model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        print("BN running mean:", m.running_mean[:5])
        print("BN running var:", m.running_var[:5])

# --------------------
# API endpoint
# --------------------
@app.post("/upload")
async def upload_data(payload: dict):
    filename = payload.get("filename")
    content = payload.get("content")

    if not filename or not content:
        raise HTTPException(status_code=400, detail="Invalid data")

    # Sauvegarde du fichier uploadé
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
        if not content.endswith("\n"):
            f.write("\n")

    # Prétraitement
    df = pre_traitement(filepath)
    features_cols = ['D1U1', 'D1U2', 'U1U2', 'D1D2', 'U1D2', 'D1U3']
    X_tensor = prepare_data_emotion_user_sequence(df, features_cols)

    # Normalisation avec les stats d'entraînement
    X_tensor = standardize_data([X_tensor])[0]  # retourne un seul tensor normalisé
    new_input = X_tensor.unsqueeze(0).to(device)  # shape: (1, 50, 6)

    # Debug info
    print("Shape new_input:", new_input.shape)
    print("Min:", torch.min(new_input).item(), "Max:", torch.max(new_input).item())
    print("Contains NaN:", torch.isnan(new_input).any().item())
    print("Contains Inf:", torch.isinf(new_input).any().item())

    # Prédiction
    with torch.no_grad():
        logits = model(new_input)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    print("Probabilities:", probs.cpu().numpy())
    print("Predicted class:", preds.item())

    # Messages selon la classe
    messages = {
        0: "ANGRY EMOTION DETECTED",
        1: "HAPPY EMOTION DETECTED",
        2: "SAD EMOTION DETECTED",
        3: "CALM EMOTION DETECTED",

    }
    message = messages.get(int(preds.item()), "Unknown emotion")
    print (message)

    return {
        "filename": filename,
        "predicted_class": int(preds.item()),
        "probabilities": probs.cpu().numpy().tolist(),
        "message": message
    }
