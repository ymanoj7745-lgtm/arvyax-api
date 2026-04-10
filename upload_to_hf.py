# ================================================================
# upload_to_hf.py — Upload models to Hugging Face Hub
# Run this ONCE from your local machine before deploying.
#
# Steps:
#   1. pip install huggingface_hub
#   2. huggingface-cli login   (enter your HF token)
#   3. python upload_to_hf.py
# ================================================================

import json
import os
from huggingface_hub import HfApi, create_repo

# ── CONFIG — change these ─────────────────────────────────────
HF_USERNAME = "ymanoj7745"
REPO_NAME   = "arvyax-models"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
# ─────────────────────────────────────────────────────────────

api = HfApi()

# Create repo (private by default — change to private=False if you want public)
print(f"Creating repo: {REPO_ID}")
try:
    create_repo(REPO_ID, private=True, exist_ok=True)
    print("Repo ready.")
except Exception as e:
    print(f"Repo may already exist: {e}")

# ── Step 1: Upload TF model ───────────────────────────────────
print("\nUploading best_model_v5.keras ...")
api.upload_file(
    path_or_fileobj = "outputs/best_model_v5.keras",
    path_in_repo    = "best_model_v5.keras",
    repo_id         = REPO_ID,
)
print("Done.")

# ── Step 2: Upload fine-tuned embedder folder ─────────────────
print("\nUploading finetuned_embedder_v5/ ...")
api.upload_folder(
    folder_path  = "outputs/finetuned_embedder_v5",
    path_in_repo = "finetuned_embedder_v5",
    repo_id      = REPO_ID,
)
print("Done.")

# ── Step 3: Pickle and upload preprocessors ───────────────────
print("\nBuilding and uploading preprocessors.json ...")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv("Sample_arvyax_reflective_dataset.csv")

NUM_COLS = ["duration_min","sleep_hours","energy_level","stress_level","intensity"]
CAT_COLS = ["ambience_type","time_of_day","previous_day_mood",
            "face_emotion_hint","reflection_quality"]

num_imp = SimpleImputer(strategy="median")
df[NUM_COLS] = num_imp.fit_transform(df[NUM_COLS])
for col in CAT_COLS:
    df[col] = df[col].fillna("unknown")

le     = LabelEncoder()
le.fit(df["emotional_state"])

ohe    = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(df[CAT_COLS])

scaler = StandardScaler()
scaler.fit(df[NUM_COLS])

prep = {
    "label_classes": le.classes_.tolist(),
    "categorical_columns": CAT_COLS,
    "ohe_categories": {
        col: cats.tolist() for col, cats in zip(CAT_COLS, ohe.categories_)
    },
    "numeric_columns": NUM_COLS,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "num_imputer_statistics": num_imp.statistics_.tolist(),
}
with open("outputs/preprocessors.json", "w", encoding="utf-8") as f:
    json.dump(prep, f, indent=2)

api.upload_file(
    path_or_fileobj = "outputs/preprocessors.json",
    path_in_repo    = "preprocessors.json",
    repo_id         = REPO_ID,
)
print("Done.")

print(f"\nAll files uploaded to: https://huggingface.co/{REPO_ID}")
print("\nNext step: set HF_REPO environment variable in Railway to:")
print(f"  {REPO_ID}")
