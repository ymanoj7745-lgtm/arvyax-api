# ================================================================
# app.py — Railway deployment entry point
# Serves: FastAPI backend + static UI on the same port
# Models: downloaded from Hugging Face Hub at startup
# ================================================================

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from threading import Lock
import numpy as np
import pandas as pd
import os, time, io, json, zipfile, tempfile, shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Model registry (lazy loaded) ──────────────────────────────
_model    = None
_embedder = None
_le       = None
_ohe      = None
_scaler   = None
_cat_cols = None
_model_lock = Lock()

HF_REPO   = os.getenv("HF_REPO", "ymanoj7745/arvyax-models")
HF_TOKEN  = os.getenv("HF_TOKEN", "")   # set in Railway env vars

def strip_quantization_config(value):
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "quantization_config" and item is None:
                continue
            cleaned[key] = strip_quantization_config(item)
        return cleaned
    if isinstance(value, list):
        return [strip_quantization_config(item) for item in value]
    return value

def sanitize_keras_archive(model_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(model_path, "r") as src:
            src.extractall(tmpdir)

        config_path = os.path.join(tmpdir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            cleaned = strip_quantization_config(config)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f)

        sanitized_path = os.path.join(tempfile.gettempdir(), "best_model_v5_sanitized.keras")
        if os.path.exists(sanitized_path):
            os.remove(sanitized_path)
        shutil.make_archive(sanitized_path[:-6], "zip", tmpdir)
        if os.path.exists(sanitized_path):
            os.remove(sanitized_path)
        os.replace(sanitized_path[:-6] + ".zip", sanitized_path)
        return sanitized_path

def encode_categories(cat_vals):
    encoded = []
    for value, col in zip(cat_vals, _cat_cols):
        categories = _ohe[col]
        row = [0.0] * len(categories)
        if value in categories:
            row[categories.index(value)] = 1.0
        encoded.extend(row)
    return np.array([encoded], dtype=np.float32)

def scale_numeric(num_vals):
    arr = np.array(num_vals, dtype=np.float32)
    means = np.array(_scaler["mean"], dtype=np.float32)
    scales = np.array(_scaler["scale"], dtype=np.float32)
    safe_scales = np.where(scales == 0, 1.0, scales)
    return ((arr - means) / safe_scales).astype(np.float32)

def load_models():
    global _model, _embedder, _le, _ohe, _scaler, _cat_cols

    if _model is not None:
        return

    import tensorflow as tf
    import keras
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import hf_hub_download, snapshot_download

    print("Downloading models from Hugging Face Hub ...")

    # Download TF classifier
    model_path = hf_hub_download(
        repo_id   = HF_REPO,
        filename  = "best_model_v5.keras",
        token     = HF_TOKEN or None,
        local_dir = "/tmp/models"
    )
    sanitized_model_path = sanitize_keras_archive(model_path)
    _model = keras.models.load_model(sanitized_model_path, compile=False)
    print("TF model loaded.")

    # Download fine-tuned sentence transformer
    embedder_dir = snapshot_download(
        repo_id   = HF_REPO,
        token     = HF_TOKEN or None,
        local_dir = "/tmp/embedder",
        ignore_patterns=["*.keras", "*.pkl", "*.csv"]
    )
    embedder_path = os.path.join(embedder_dir, "finetuned_embedder_v5")
    if not os.path.exists(embedder_path):
        embedder_path = embedder_dir
    _embedder = SentenceTransformer(embedder_path)
    print("Embedder loaded.")

    # Download preprocessing metadata
    prep_path = hf_hub_download(
        repo_id   = HF_REPO,
        filename  = "preprocessors.json",
        token     = HF_TOKEN or None,
        local_dir = "/tmp/models"
    )
    with open(prep_path, "r", encoding="utf-8") as f:
        prep = json.load(f)

    _le = prep["label_classes"]
    _ohe = prep["ohe_categories"]
    _scaler = {
        "mean": prep["scaler_mean"],
        "scale": prep["scaler_scale"],
    }
    _cat_cols = prep["categorical_columns"]
    print("Preprocessors loaded. System ready.")

def ensure_models_loaded():
    if _model is not None:
        return
    with _model_lock:
        if _model is None:
            load_models()


# ── Decision engine ───────────────────────────────────────────
def intensity_band(i): return "low" if i<=2 else "mid" if i==3 else "high"
def energy_band(e):    return "low" if e<=2 else "mid" if e==3 else "high"
def time_band(t):
    return ("morning" if t in ("morning","early_morning")
            else "afternoon" if t=="afternoon"
            else "evening"  if t=="evening" else "night")

def decide(state, intensity, stress, energy, time_of_day, uncertain=False):
    ib, eb, tb = intensity_band(intensity), energy_band(energy), time_band(time_of_day)
    hi_stress  = stress >= 4
    if state == "overwhelmed":
        if ib=="high" or hi_stress: action,timing,conf="grounding","now",0.92
        elif ib=="mid":             action,timing,conf="box_breathing","now",0.88
        else:                       action,timing,conf="light_planning","within_15_min",0.80
    elif state == "restless":
        if eb=="high":                     action,timing,conf="movement","now",0.90
        elif tb in("morning","afternoon"): action,timing,conf="deep_work","within_15_min",0.82
        else:                              action,timing,conf="yoga","within_15_min",0.78
    elif state == "mixed":
        if hi_stress:   action,timing,conf="box_breathing","now",0.85
        elif ib=="high":action,timing,conf="journaling","within_15_min",0.83
        else:           action,timing,conf="pause","now",0.75
    elif state == "focused":
        if tb in("morning","afternoon") and eb in("mid","high"):
            action,timing,conf="deep_work","now",0.95
        elif tb=="evening": action,timing,conf="light_planning","within_15_min",0.84
        else:               action,timing,conf="journaling","within_15_min",0.78
    elif state == "calm":
        if tb=="morning" and eb in("mid","high"): action,timing,conf="deep_work","now",0.92
        elif tb=="night" or eb=="low":            action,timing,conf="rest","now",0.90
        elif ib=="high":                          action,timing,conf="yoga","within_15_min",0.80
        else:                                     action,timing,conf="sound_therapy","later_today",0.72
    elif state == "neutral":
        if tb in("morning","afternoon") and eb in("mid","high"):
            action,timing,conf="deep_work","within_15_min",0.82
        elif hi_stress: action,timing,conf="movement","within_15_min",0.80
        elif tb in("evening","night"): action,timing,conf="journaling","tonight",0.76
        else:           action,timing,conf="light_planning","within_15_min",0.70
    else:
        action,timing,conf="pause","now",0.50
    if tb=="night" and action in("deep_work","movement","yoga"):
        action,timing,conf="rest","tonight",min(conf,0.80)
    if uncertain:
        conf = max(conf*0.85, 0.50)
    return action, timing, round(conf, 3)

def supportive_message(state, action, intensity):
    messages = {
        ("overwhelmed","grounding"):      "You're carrying a lot right now. Let's bring you back to this moment — just 5 minutes of grounding can make a real difference.",
        ("overwhelmed","box_breathing"):  "Take a breath. Your nervous system just needs a reset. Box breathing will help you find your footing again.",
        ("overwhelmed","light_planning"): "It sounds like there's a lot in your head. Let's get it out on paper — even a small plan can quiet the mental noise.",
        ("restless","movement"):          "That energy needs somewhere to go. A quick movement session will help you feel clearer and more settled.",
        ("restless","deep_work"):         "Your mind is buzzing — that can actually be a superpower. Let's channel it into something meaningful.",
        ("restless","yoga"):              "Your body is telling you something. Yoga will help you listen and settle into the evening.",
        ("mixed","box_breathing"):        "It's okay not to have it all figured out. Let's start with your breath and let things become clearer from there.",
        ("mixed","journaling"):           "When feelings are tangled, writing helps untangle them. No pressure — just let the words come.",
        ("focused","deep_work"):          "You're in a great state right now. This is your window — protect it and make the most of this clarity.",
        ("calm","deep_work"):             "Calm and focused is the ideal state. You've got everything you need right now — let's use it.",
        ("calm","rest"):                  "Your body knows what it needs. Rest isn't giving up — it's how you show up fully tomorrow.",
        ("neutral","deep_work"):          "Neutral is a stable place to build from. A focused session now could shift the whole trajectory of your day.",
    }
    fallbacks = {
        "journaling":     "Writing is how we make sense of what we feel. Take 10 minutes — you might surprise yourself.",
        "box_breathing":  "Your breath is always with you. Let it be your anchor right now.",
        "grounding":      "The present moment is safe. Let's come back to it together.",
        "movement":       "Movement is medicine. Even 5 minutes will shift your state.",
        "rest":           "Rest is productive. Your brain consolidates and repairs when you step back.",
        "yoga":           "Yoga meets you exactly where you are. No expectations — just presence.",
        "sound_therapy":  "Let sound do the work. Just listen and let your nervous system soften.",
        "light_planning": "Small steps forward are still forward. Let's build just a little structure.",
        "deep_work":      "You have more capacity than you think. Let's put it to good use.",
        "pause":          "Sometimes the bravest thing is to pause. You've earned a moment of stillness.",
    }
    return messages.get((state, action), fallbacks.get(action, "Take care of yourself. One step at a time."))


# ================================================================
# FASTAPI APP
# ================================================================
app = FastAPI(title="Arvyax Emotional Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
async def startup():
    import asyncio
    asyncio.create_task(load_models_async())

async def load_models_async():
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)



# ── Schemas ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    journal_text      : str
    ambience_type     : Optional[str]  = "outdoor"
    duration_min      : Optional[float]= 20.0
    sleep_hours       : Optional[float]= 7.0
    energy_level      : Optional[int]  = 3
    stress_level      : Optional[int]  = 3
    time_of_day       : Optional[str]  = "afternoon"
    previous_day_mood : Optional[str]  = "neutral"
    face_emotion_hint : Optional[str]  = "neutral_face"
    reflection_quality: Optional[str]  = "clear"
    intensity         : Optional[int]  = None

class PredictionResult(BaseModel):
    emotional_state     : str
    emotion_confidence  : float
    uncertainty_flag    : str
    intensity_predicted : float
    intensity_source    : str
    action              : str
    timing              : str
    decision_confidence : float
    supportive_message  : str
    latency_ms          : float


# ── Feature builder ───────────────────────────────────────────
def build_features(req: PredictRequest):
    cat_vals = [req.ambience_type or "unknown", req.time_of_day or "unknown",
                req.previous_day_mood or "unknown", req.face_emotion_hint or "unknown",
                req.reflection_quality or "unknown"]
    num_vals = [[req.duration_min or 20.0, req.sleep_hours or 7.0,
                 req.energy_level or 3, req.stress_level or 3,
                 req.intensity or 3]]
    emb = _embedder.encode([req.journal_text], normalize_embeddings=True)
    cat = encode_categories(cat_vals)
    num = scale_numeric(num_vals)
    return np.hstack([emb, cat, num]).astype(np.float32)

def get_uncertainty(proba):
    eps     = 1e-9
    conf    = float(np.max(proba))
    entropy = float(-np.sum(proba * np.log(proba + eps)))
    if conf >= 0.60 and entropy <= 1.2:  return "confident"
    elif conf < 0.50 or entropy > 1.5:   return "uncertain"
    else:                                 return "borderline"


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": _model is not None,
        "hf_repo": HF_REPO,
    }

@app.get("/classes")
def get_classes():
    ensure_models_loaded()
    return {
        "emotional_states": list(_le) if _le else [],
        "actions": ["box_breathing","journaling","grounding","deep_work","yoga",
                    "sound_therapy","light_planning","rest","movement","pause"],
        "timings": ["now","within_15_min","later_today","tonight","tomorrow_morning"]
    }

@app.post("/predict", response_model=PredictionResult)
def predict(req: PredictRequest):
    ensure_models_loaded()
    if _model is None:
        raise HTTPException(503, "Models failed to load")
    t0 = time.time()

    word_count = len(req.journal_text.split())
    X          = build_features(req)
    proba      = _model.predict(X, verbose=0)[0]
    pred_idx   = int(np.argmax(proba))
    pred_state = _le[pred_idx]
    conf       = float(np.max(proba))
    unc_flag   = get_uncertainty(proba)

    if word_count <= 3:
        unc_flag = "uncertain"
        conf     = conf * 0.7

    if req.intensity is not None:
        intensity_val    = float(req.intensity)
        intensity_source = "provided"
    else:
        stress        = req.stress_level or 3
        energy        = req.energy_level or 3
        intensity_val = round(np.clip((stress + (6 - energy)) / 2, 1, 5), 2)
        intensity_source = "estimated"

    action, timing, dec_conf = decide(
        pred_state, intensity_val,
        req.stress_level or 3, req.energy_level or 3,
        req.time_of_day  or "afternoon",
        uncertain = (unc_flag == "uncertain")
    )

    message = supportive_message(pred_state, action, intensity_val)
    latency = round((time.time() - t0) * 1000, 1)

    return PredictionResult(
        emotional_state     = pred_state,
        emotion_confidence  = round(conf, 4),
        uncertainty_flag    = unc_flag,
        intensity_predicted = intensity_val,
        intensity_source    = intensity_source,
        action              = action,
        timing              = timing,
        decision_confidence = dec_conf,
        supportive_message  = message,
        latency_ms          = latency
    )


# ── Serve UI ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = "bonus3_ui.html"
    if os.path.exists(ui_path):
        with open(ui_path, "r") as f:
            html = f.read()
        # Replace localhost API URL with relative path for production
        html = html.replace(
            'const API = "http://localhost:8000"',
            'const API = ""'
        )
        return HTMLResponse(html)
    return HTMLResponse("<h2>UI not found. API is running at /docs</h2>")
