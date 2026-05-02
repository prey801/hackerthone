import re
import numpy as np

CONFIDENCE_THRESHOLD = 0.35  # minimum retrieval score to attempt an LLM response

def assess_risk(issue_text, domain, max_retrieval_score, embedder=None):
    """
    Rule-based and semantic escalation triggers.
    Returns: {should_escalate: bool, reason: str, risk_level: str}
    """
    normalized_issue = str(issue_text).lower()

    # Trigger Categories
    risk_patterns = {
        "financial_fraud": ["unauthorized charge", "fraudulent", "stolen card", "dispute", "stolen in", "lost my card", "fraud"],
        "account_access": ["locked out", "can't login", "account banned", "suspended"],
        "assessment_integrity": ["cheating", "plagiarism", "unfair result", "contest fraud", "leaked"],
        "legal_privacy": ["gdpr", "data deletion", "legal action", "lawsuit", "delete my data"],
        "billing_disputes": ["refund", "wrong charge", "double charged"],
        "security": ["hacked", "breach", "phishing", "compromised"]
    }

    # 1. Exact regex matching
    for category, patterns in risk_patterns.items():
        for pattern in patterns:
            if re.search(rf"\b{re.escape(pattern)}\b", normalized_issue):
                return {
                    "should_escalate": True, 
                    "reason": f"Risk triggered: {category} ({pattern})", 
                    "risk_level": "high",
                    "risk_category": category
                }

    # 2. Semantic matching fallback
    if embedder is not None:
        risk_descriptions = {
            "financial_fraud": "Unauthorized credit card charge, fraudulent activity, stolen card, disputed transaction.",
            "account_access": "Account locked out, suspended user, banned profile, unable to login securely.",
            "assessment_integrity": "Candidate cheating, code plagiarism, leaked test questions, unfair assessment result.",
            "legal_privacy": "GDPR compliance, legal lawsuit action, delete my personal data entirely.",
            "billing_disputes": "Refund request, double charged on statement, incorrect billing amount.",
            "security": "System breached, hacked account, phishing attempt, security compromised."
        }
        categories = list(risk_descriptions.keys())
        descriptions = list(risk_descriptions.values())
        
        issue_emb = embedder.encode([normalized_issue])[0]
        desc_embs = embedder.encode(descriptions)
        
        issue_norm = issue_emb / np.linalg.norm(issue_emb)
        desc_norms = desc_embs / np.linalg.norm(desc_embs, axis=1)[:, np.newaxis]
        similarities = np.dot(desc_norms, issue_norm)
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score > 0.45: # Higher threshold for risk escalation
            category = categories[best_idx]
            return {
                "should_escalate": True, 
                "reason": f"Semantic risk triggered: {category} (score: {best_score:.2f})", 
                "risk_level": "high",
                "risk_category": category
            }

    # 3. Check retrieval confidence
    if max_retrieval_score < CONFIDENCE_THRESHOLD:
        return {
            "should_escalate": True,
            "reason": f"No corpus coverage for this issue (max_score={max_retrieval_score:.2f} < {CONFIDENCE_THRESHOLD})",
            "risk_level": "medium",
            "risk_category": "low_confidence"
        }

    return {"should_escalate": False, "reason": "No risks detected", "risk_level": "low", "risk_category": None}
