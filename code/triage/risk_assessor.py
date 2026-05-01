def assess_risk(issue_text, domain, max_retrieval_score):
    """
    Rule-based escalation triggers.
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
    
    for category, patterns in risk_patterns.items():
        for pattern in patterns:
            if pattern in normalized_issue:
                return {"should_escalate": True, "reason": f"Risk triggered: {category} ({pattern})", "risk_level": "high"}
                
    # Check retrieval confidence
    if max_retrieval_score < 0.35:
        return {"should_escalate": True, "reason": f"Low retrieval confidence ({max_retrieval_score:.2f})", "risk_level": "medium"}
        
    return {"should_escalate": False, "reason": "No risks detected", "risk_level": "low"}
