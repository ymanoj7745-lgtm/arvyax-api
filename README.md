# <img src="https://img.shields.io/badge/🧠-Arvyax-6C63FF?style=flat-square&labelColor=1a1a2e" height="32"/> Arvyax Emotional Intelligence API

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Model%20Hub-FFD21E?style=for-the-badge)
![Render](https://img.shields.io/badge/Render-Web%20Service-46E3B7?style=for-the-badge&logo=render&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.2-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

**Predict emotional states from journal entries · Get personalized action recommendations · Deploy anywhere**

[🚀 Live Demo](https://arvyax-api-ibfu.onrender.com) · [📖 API Docs](https://arvyax-api-ibfu.onrender.com/docs) · [🤗 Models](https://huggingface.co/ymanoj7745/arvyax-models) · [📊 Dataset](https://huggingface.co/datasets)

</div>

---

## 🧠 What is Arvyax?

Arvyax is an **end-to-end Emotional Intelligence system** that reads a user's journal entry, understands their emotional context, and delivers a precise, personalized wellness recommendation — *what to do and when to do it*.

```
📝 "The forest session helped me feel clearer, but I keep drifting to tomorrow's tasks."
                              ↓
              🔍 Emotional State:  focused  (conf: 0.74)
              ⚡ Intensity:         3.2 / 5
              ✅ Action:           deep_work
              ⏰ When:             now
              💬 "You're in a great state right now. This is your window — protect it."
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **6-Class Emotion Detection** | calm · focused · mixed · neutral · overwhelmed · restless |
| 📊 **Intensity Prediction** | Ordinal 1–5 scale with regression + classification heads |
| 🤖 **Decision Engine** | Rule-based action recommender (10 actions × 5 timings) |
| 🔍 **Uncertainty Quantification** | Entropy-based confidence flagging |
| 🧹 **Label Noise Handling** | Confident learning (Bonus 1) |
| 🌐 **REST API** | FastAPI with Swagger UI |
| 🎨 **Interactive UI** | Beautiful single-page journaling interface |
| 🚀 **Production Ready** | Publicly deployed on Render with model assets on Hugging Face |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                              │
│   journal_text  +  ambience  +  sleep  +  stress  +  energy │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    FEATURE ENGINEERING   │
          │  ┌──────────────────┐   │
          │  │ TF-IDF (250 dim) │   │  ← Text features
          │  │ OHE Categorical  │   │  ← Context features
          │  │ Scaled Numerics  │   │  ← Biometric features
          │  │ 13 Interactions  │   │  ← Domain knowledge
          │  └──────────────────┘   │
          │     295 total features   │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │    EMOTION ENSEMBLE      │
          │                          │
          │  RF (35%) ────────┐      │
          │  HGB (45%) ───────┼──▶  │  Soft voting
          │  LogReg (20%) ────┘      │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   INTENSITY REGRESSOR    │
          │   RandomForest → 1–5     │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │     DECISION ENGINE      │
          │  state × intensity ×     │
          │  energy × time_of_day    │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │       OUTPUT             │
          │  predicted_state         │
          │  predicted_intensity     │
          │  confidence              │
          │  uncertain_flag          │
          │  what_to_do              │
          │  when_to_do              │
          └─────────────────────────┘
```

---

## 📊 Dataset Overview

| Attribute | Value |
|---|---|
| Training samples | 1,200 |
| Test samples | 120 |
| Target classes | 6 (balanced) |
| Text features | Journal entries (avg 12 words) |
| Numeric features | duration, sleep, energy, stress, intensity |
| Categorical features | ambience, time_of_day, prev_mood, face_hint, reflection_quality |

**Class Distribution:**

```
calm        ████████████████████  216  (18.0%)
restless    ███████████████████   209  (17.4%)
neutral     ██████████████████    201  (16.8%)
focused     █████████████████     193  (16.1%)
mixed       █████████████████     191  (15.9%)
overwhelmed ████████████████      190  (15.8%)
```

---

## 🔬 Feature Engineering

```python
# 13 hand-crafted domain features
stress_energy_ratio   = stress / (energy + 0.1)      # activation index
sleep_deficit         = 8.0 - sleep_hours             # recovery debt
stress_sleep_interact = stress * sleep_deficit        # compound burden
wellbeing_score       = sleep + energy - stress       # composite health
sentiment_proxy       = pos_word_count - neg_word_count  # text valence
# + text_len, word_count, excl_count, quest_count, neg/pos word counts
```

---

## 🤖 Model Performance

### Emotion State Classification

| Model | Val Accuracy | Val F1 |
|---|---|---|
| Random Forest | 57.9% | 0.578 |
| HistGradientBoosting | 50.8% | 0.507 |
| Logistic Regression | 49.6% | 0.495 |
| **Ensemble (RF+HGB+LR)** | **53.3%** | **0.534** |
| TF v5 + MPNet (full) | ~72%+ | ~0.72+ |

> ℹ️ 53% on 6-class balanced problem = **3.2× better than random chance** (16.7%). The full TF v5 model with fine-tuned MPNet achieves 72%+ — see `emotional_state_tensorflow_v5.py`.

### Per-Class Performance

```
calm        P: 0.59  R: 0.60  F1: 0.60  ← Best class
overwhelmed P: 0.55  R: 0.63  F1: 0.59
mixed       P: 0.67  R: 0.53  F1: 0.59
focused     P: 0.43  R: 0.51  F1: 0.47
neutral     P: 0.53  R: 0.45  F1: 0.49
restless    P: 0.38  R: 0.38  F1: 0.38  ← Hardest class
```

---

## 🎯 Decision Engine

The rule-based engine maps `(state × intensity × energy × time_of_day)` to actions:

```
overwhelmed + high_intensity  ──▶  grounding      (now)
overwhelmed + mid_intensity   ──▶  box_breathing  (now)
restless    + high_energy     ──▶  movement       (now)
restless    + morning         ──▶  deep_work      (within 15min)
focused     + morning         ──▶  deep_work      (now)
calm        + night           ──▶  rest           (now)
neutral     + hi_stress       ──▶  movement       (within 15min)
```

**Night override**: `deep_work`, `movement`, `yoga` → replaced with `rest` after dark.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Run the pipeline
```bash
git clone https://github.com/ymanoj7745-lgtm/arvyax-api
cd arvyax-api
python pipeline.py
```

### Run the full API locally
```bash
pip install -r requirements.txt
# Set environment variables
export HF_REPO="ymanoj7745/arvyax-models"
export HF_TOKEN="your_hf_token"
# Start server
uvicorn app:app --reload --port 8000
```

Then open `http://localhost:8000` for the UI or `http://localhost:8000/docs` for API docs.

---

## 📡 API Reference

### `POST /predict`

```json
{
  "journal_text": "The forest session helped me feel clearer today.",
  "ambience_type": "forest",
  "sleep_hours": 7.0,
  "energy_level": 4,
  "stress_level": 2,
  "time_of_day": "morning",
  "previous_day_mood": "neutral",
  "reflection_quality": "clear"
}
```

**Response:**
```json
{
  "emotional_state": "focused",
  "emotion_confidence": 0.7423,
  "uncertainty_flag": "confident",
  "intensity_predicted": 2.8,
  "intensity_source": "estimated",
  "action": "deep_work",
  "timing": "now",
  "decision_confidence": 0.95,
  "supportive_message": "You're in a great state right now. This is your window.",
  "latency_ms": 18.4
}
```

### Other Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Interactive UI |
| `/health` | GET | Health check + model status |
| `/classes` | GET | Available states, actions, timings |
| `/predict` | POST | Main prediction endpoint |
| `/docs` | GET | Swagger UI |

---

## 📁 Project Structure

```
arvyax-api/
│
├── 📄 pipeline.py                        # End-to-end sklearn pipeline
├── 📄 app.py                             # FastAPI deployment entry point
├── 📄 emotional_state_tensorflow_v5.py  # Full TF v5 model
├── 📄 part2_intensity.py                # Intensity prediction
├── 📄 part3_decision_engine.py          # Decision engine
├── 📄 parts4to9_analysis.py             # Analysis scripts
├── 📄 bonus1_label_noise.py             # Confident learning
├── 📄 bonus2_api.py                     # API bonus
├── 📄 bonus3_ui.html                    # Interactive UI
│
├── 📊 predictions.csv                   # Test predictions (120 rows)
├── 📊 Sample_arvyax_reflective_dataset.csv  # Training data
├── 📊 arvyax_test_inputs.csv            # Test inputs
│
├── 📖 README.md                         # This file
├── 📖 ERROR_ANALYSIS.md                 # 10 failure cases + insights
├── 📖 EDGE_PLAN.md                      # Deployment + optimization plan
│
├── 🐳 Dockerfile                        # Container deployment
├── 📋 requirements.txt                  # Dependencies
└── 📋 Procfile                          # Process startup config
```

---

## 📋 Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | End-to-end pipeline (`pipeline.py`) | ✅ |
| 2 | `predictions.csv` (id, state, intensity, confidence, flag, action, timing) | ✅ |
| 3 | `README.md` (setup, approach, features, model, how to run) | ✅ |
| 4 | `ERROR_ANALYSIS.md` (10 failure cases + insights) | ✅ |
| 5 | `EDGE_PLAN.md` (deployment + optimizations) | ✅ |
| B1 | Label noise handling (confident learning) | ✅ |
| B2 | REST API with FastAPI | ✅ |
| B3 | Interactive UI | ✅ |

---

## 🌐 Deployment

The public app is deployed on **Render**, while model artifacts are hosted on **Hugging Face Hub**:

```
User Request
    ↓
Render Web Service (FastAPI app)
    ↓
Download models from HF Hub on startup / first load
    ├── best_model_v5.keras
    ├── finetuned_embedder_v5/
    └── preprocessors.json
    ↓
Serve predictions + UI
```

**Live URLs:**
- App: https://arvyax-api-ibfu.onrender.com
- API: https://arvyax-api-ibfu.onrender.com/docs
- Models: https://huggingface.co/ymanoj7745/arvyax-models

---

## 🔮 Future Work

- [ ] Replace TF-IDF with full MPNet embeddings in sklearn pipeline
- [ ] Add contrast conjunction features for `mixed` state recall
- [ ] Implement adaptive class-conditional confidence thresholds
- [ ] Mobile app with TFLite quantized model (~8MB)
- [ ] Weekly retraining with user feedback corrections
- [ ] Multilingual journal support

---

## 👨‍💻 Author

**Manoj** · [@ymanoj7745](https://huggingface.co/ymanoj7745)

---

<div align="center">

Made with ❤️ · Powered by Hugging Face Hub · Built with FastAPI

</div>
