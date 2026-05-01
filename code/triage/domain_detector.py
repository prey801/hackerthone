def detect_domain(company_field, issue_text, retriever=None):
    """
    Priority order:
    1. If company in {HackerRank, Claude, Visa} -> use directly
    2. If company = None or 'none': infer from keyword signals
    """
    valid_domains = {"hackerrank", "claude", "visa"}
    
    # 1. Direct match (also handle 'none' as a string from CSV)
    if company_field:
        normalized_company = str(company_field).strip().lower()
        if normalized_company in valid_domains:
            return normalized_company, 1.0
        # 'none' string or 'nan' from pandas -> fall through to inference
            
    # 2. Keyword signals fallback
    normalized_issue = str(issue_text).lower()
    
    visa_terms = [
        "card", "payment", "cheque", "stolen card", "fraud", "charge", "bank",
        "visa", "refund", "merchant", "transaction", "cash", "atm", "travel",
        "spending", "currency", "minimum spend"
    ]
    hr_terms = [
        "test", "assessment", "code", "runtime", "compiler", "candidate", "interview",
        "hackerrank", "submission", "challenge", "hiring", "recruiter", "score",
        "certificate", "proctoring", "zoom", "resume", "mock", "subscription"
    ]
    claude_terms = [
        "claude", "chat", "conversation", "ai", "model", "prompt", "token", "message",
        "api", "bedrock", "workspace", "lti", "data training", "crawl", "bug bounty",
        "requests failing", "security vulnerability"
    ]
    
    scores = {"hackerrank": 0, "claude": 0, "visa": 0}
    
    for term in visa_terms:
        if term in normalized_issue: scores["visa"] += 1
    for term in hr_terms:
        if term in normalized_issue: scores["hackerrank"] += 1
    for term in claude_terms:
        if term in normalized_issue: scores["claude"] += 1
        
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] > 0:
        return best_domain, 0.8
        
    return None, 0.0
