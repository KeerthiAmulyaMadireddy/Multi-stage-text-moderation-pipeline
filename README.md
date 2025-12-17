# Multi-Stage Text Moderation Engine

A production-ready **multi-stage text moderation system** designed for real-time chat platforms.
The engine combines **rule-based (regex/lexicon) filtering** with a **machine-learning classifier**
to detect harmful, abusive, explicit, violent, self-harm, scam, and spam content.

This project is built to be:
- Explainable
- Fast (Stage-0 rules first)
- Extensible (new terms, new ML models)
- Production-friendly (penalty scoring, API-ready)

---

## ⚠️ Disclaimer

This repository contains **offensive, hateful, violent, and explicit terms**  
**strictly for research, safety, and content-moderation purposes.**

These terms are:
- **NOT endorsed**
- **NOT encouraged**
- **NOT used in any harmful context**

They exist **only** to help detect, block, and prevent abusive behavior
in chat platforms and user-generated content systems.

---

## 🧠 Moderation Architecture

### Stage-0: Rule-Based (Regex & Lexicon)
- High-confidence detection
- Covers:
  - Hate & harassment
  - Sexual content (explicit, soft, minors)
  - Violence & threats
  - Self-harm & suicide ideation
  - Spam & scam attempts
- Uses:
  - Regex patterns
  - Phrase matching
  - Leetspeak normalization (e.g., `stup1d → stupid`)
- Immediate action for critical violations

### Stage-1: Machine Learning (TF-IDF + Logistic Regression)
- Multi-label classifier trained on public datasets
- Detects nuanced toxicity where rules don’t apply
- Labels:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate
- Uses per-label probability thresholds

### Penalty Scoring
- Assigns a numeric penalty per violation
- Combines:
  - Stage-0 severity
  - Stage-1 ML confidence
- Designed to support:
  - User warnings
  - Rate limiting
  - Account flagging
  - Trust & safety workflows

---

## 📂 Project Structure

```text
.
├── moderation_engine.py   # Core moderation logic (Stage-0 + Stage-1 + penalties)
├── api.py                 # FastAPI endpoint for real-time moderation
├── train_model.py         # ML training script (TF-IDF + Logistic Regression)
├── moderation_model.pkl   # Trained ML model (generated locally)
├── terms/
│   └── terms.json         # Moderation lexicon (regex + word lists)
