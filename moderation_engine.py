import json
import re
from typing import Dict, List, Optional, Tuple

import joblib

# Load Stage-0 Terms (lexicon from JSON)
def load_terms(path: str = "terms/terms.json") -> Dict:
    """Load moderation lexicon from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


TERMS = load_terms()

LEET_MAP = str.maketrans({
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i"
})


def normalize_text(text: str) -> str:
    """
    Lowercase, strip whitespace, normalize leetspeak,
    and collapse multiple spaces.
    """
    t = text.lower()
    t = t.translate(LEET_MAP)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Compile regex patterns for Stage-0
def compile_word_list(words: List[str]) -> List[re.Pattern]:
    """
    Compile a list of words/phrases into case-insensitive regex patterns.
    Single words get word boundaries; phrases match anywhere.
    """
    patterns = []
    for w in words:
        escaped = re.escape(w)
        if " " in w:
            # Phrase, match anywhere in the text
            pattern = re.compile(escaped, re.IGNORECASE)
        else:
            # Single word, match with word boundaries
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        patterns.append(pattern)
    return patterns


def compile_regex_list(regexes: List[str]) -> List[re.Pattern]:
    """Compile a list of raw regex patterns."""
    return [re.compile(r, re.IGNORECASE) for r in regexes]


# Map categories to compiled patterns
CATEGORY_PATTERNS: Dict[str, List[re.Pattern]] = {
    "hate": compile_word_list(TERMS.get("hate_terms", [])),
    "harassment": compile_word_list(TERMS.get("harassment_terms", [])),

    "sexual_explicit": compile_word_list(TERMS.get("sexual_terms_explicit", [])),
    "sexual_soft": compile_word_list(TERMS.get("sexual_terms_soft", [])),
    "sexual_minors": compile_word_list(TERMS.get("sexual_minors_terms", [])),

    "violence": compile_word_list(TERMS.get("violence_terms", [])),
    "violence_pattern": compile_regex_list(TERMS.get("violence_patterns", [])),

    "self_harm": compile_word_list(TERMS.get("self_harm_terms", [])),
    "self_harm_pattern": compile_regex_list(TERMS.get("self_harm_patterns", [])),

    "spam": compile_regex_list(TERMS.get("spam_patterns", [])),
    "scam": compile_regex_list(TERMS.get("scam_patterns", [])),
}

# Severity & Action for Stage-0 categories (Balanced mode)
CATEGORY_SEVERITY_ACTION: Dict[str, Tuple[str, str]] = {
    "sexual_minors": ("critical", "block_and_alert"),
    "self_harm": ("critical", "block_and_alert"),
    "self_harm_pattern": ("critical", "block_and_alert"),

    "violence": ("high", "block"),
    "violence_pattern": ("high", "block"),

    "hate": ("high", "block"),
    "harassment": ("medium", "warn"),

    "sexual_explicit": ("medium", "warn"),
    "sexual_soft": ("low", "warn"),

    "scam": ("medium", "warn"),
    "spam": ("low", "warn"),
}

# Priority order: first match at top wins
STAGE0_PRIORITY = [
    "sexual_minors",
    "self_harm_pattern",
    "self_harm",
    "violence_pattern",
    "violence",
    "hate",
    "harassment",
    "sexual_explicit",
    "sexual_soft",
    "scam",
    "spam",
]

# Stage-0 main function
def stage0_check(text: str) -> Optional[Dict]:
    """
    Run Stage-0 regex rules on the text.
    Returns a violation dict if something is caught, else None.
    """
    norm = normalize_text(text)
    hits: List[Tuple[str, str]] = []

    for category, patterns in CATEGORY_PATTERNS.items():
        for p in patterns:
            if p.search(norm):
                hits.append((category, p.pattern))

    if not hits:
        return None

    # Group patterns per category
    hits_by_cat: Dict[str, List[str]] = {}
    for cat, pat in hits:
        hits_by_cat.setdefault(cat, []).append(pat)

    # Pick highest-priority category
    chosen_cat = None
    for cat in STAGE0_PRIORITY:
        if cat in hits_by_cat:
            chosen_cat = cat
            break

    if chosen_cat is None:
        # Fallback (shouldn't happen)
        chosen_cat = hits[0][0]

    severity, action = CATEGORY_SEVERITY_ACTION.get(
        chosen_cat, ("medium", "warn")
    )

    return {
        "stage": "stage0",
        "category": chosen_cat,
        "severity": severity,
        "action": action,
        "matches": hits_by_cat[chosen_cat],
    }


# Stage-1 ML Model (TF-IDF + Logistic Regression)
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Thresholds for each label (will tune later based on ND data)
THRESHOLDS: Dict[str, float] = {
    "toxic": 0.9,         
    "severe_toxic": 0.7,  
    "obscene": 0.8,        
    "threat": 0.7,        
    "insult": 0.9,         
    "identity_hate": 0.7, 
}


class Stage1Model:
    def __init__(self, model_path: str = "moderation_model.pkl"):
        print(f"[Stage1Model] Loading model from: {model_path}")
        self.model = joblib.load(model_path)

    def predict(self, text: str):
        probs = self.model.predict_proba([text])[0]
        scores = {label: float(probs[i]) for i, label in enumerate(LABELS)}

        flagged = [
            label for label, prob in scores.items()
            if prob >= THRESHOLDS[label]
        ]
        return scores, flagged


# Penalty scoring (Stage-0 + Stage-1)

STAGE0_PENALTY: Dict[str, int] = {
    "sexual_minors": 15,
    "self_harm": 12,
    "self_harm_pattern": 12,
    "violence": 12,
    "violence_pattern": 12,
    "hate": 10,
    "harassment": 5,
    "sexual_explicit": 6,
    "sexual_soft": 3,
    "scam": 4,
    "spam": 2,
}

STAGE1_PENALTY: Dict[str, int] = {
    "severe_toxic": 8,
    "threat": 10,
    "identity_hate": 7,
    "toxic": 5,
    "insult": 3,
    "obscene": 3,
}


def compute_penalty(stage0_category: Optional[str] = None,
                    stage1_labels: Optional[List[str]] = None) -> Tuple[int, List[str]]:
    """
    Compute total penalty and explanation list.
    """
    total = 0
    reasons: List[str] = []

    # Stage-0 penalty
    if stage0_category:
        pts = STAGE0_PENALTY.get(stage0_category, 0)
        total += pts
        reasons.append(f"stage0:{stage0_category}(+{pts})")

    # Stage-1 penalty
    if stage1_labels:
        for lbl in stage1_labels:
            pts = STAGE1_PENALTY.get(lbl, 0)
            total += pts
            reasons.append(f"stage1:{lbl}(+{pts})")

    return total, reasons

# Final Moderation Engine
class ModerationEngine:
    """
    Orchestrates:
      Stage-0 -> Stage-1 -> Penalty and produces a final moderation decision.
    """
    def __init__(self, model_path: str = "moderation_model.pkl"):
        self.stage1 = Stage1Model(model_path)

    def moderate(self, text: str) -> Dict:
        # 1) Stage-0: rules / regex
        s0 = stage0_check(text)
        if s0:
            penalty_score, reasons = compute_penalty(stage0_category=s0["category"])
            return {
                "stage": "stage0",
                "categories": [s0["category"]],
                "severity": s0["severity"],
                "action": s0["action"],
                "penalty_score": penalty_score,
                "penalty_reasons": reasons,
                "details": {
                    "regex_matches": s0["matches"],
                    "ml_scores": None,
                },
            }

        # 2) Stage-1: ML model (only if Stage-0 is clean)
        scores, flagged = self.stage1.predict(text)

        if not flagged:
            # Completely clean
            return {
                "stage": "clean",
                "categories": [],
                "severity": "none",
                "action": "allow",
                "penalty_score": 0,
                "penalty_reasons": [],
                "details": {
                    "regex_matches": [],
                    "ml_scores": scores,
                },
            }

        # Decide severity / action from ML labels
        severity = "low"
        action = "allow"

        penalty_score, reasons = compute_penalty(stage1_labels=flagged)

        return {
            "stage": "stage1",
            "categories": flagged,
            "severity": severity,
            "action": action,
            "penalty_score": penalty_score,
            "penalty_reasons": reasons,
            "details": {
                "regex_matches": [],
                "ml_scores": scores,
            },
        }


# CLI Test (run: python moderation_engine.py)
if __name__ == "__main__":
    engine = ModerationEngine("moderation_model.pkl")
    print("🔍 Text Moderation Console (type 'exit' to quit)")
    while True:
        msg = input("\nUser message: ")
        if msg.lower().strip() == "exit":
            break
        result = engine.moderate(msg)
        print(result)