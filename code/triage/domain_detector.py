import numpy as np

def detect_domain(company_field, issue_text, retriever=None, embedder=None):
    """
    Priority order:
    1. If company in {HackerRank, Claude, Visa} -> use directly
    2. If company = None or 'none': infer from keyword signals and semantic similarity
    """
    valid_domains = {"hackerrank", "claude", "visa"}
    
    # 1. Direct match (also handle 'none' as a string from CSV)
    if company_field:
        normalized_company = str(company_field).strip().lower()
        if normalized_company in valid_domains:
            return normalized_company, 1.0
            
    normalized_issue = str(issue_text).lower()
    
    # Semantic similarity fallback
    if embedder is not None:
        domain_descriptions = {
            "visa": "credit card, payment processing, fraud, bank transaction, refund, merchant cheque.",
            "hackerrank": "coding test assessment, compiler runtime error, candidate interview, proctoring, submission score.",
            "claude": "AI language model, chat conversation, API token limit, bedrock workspace, prompt instructions."
        }
        domain_keys = list(domain_descriptions.keys())
        descriptions = list(domain_descriptions.values())
        
        issue_emb = embedder.encode([normalized_issue])[0]
        desc_embs = embedder.encode(descriptions)
        
        issue_norm = issue_emb / np.linalg.norm(issue_emb)
        desc_norms = desc_embs / np.linalg.norm(desc_embs, axis=1)[:, np.newaxis]
        similarities = np.dot(desc_norms, issue_norm)
        
        best_idx = np.argmax(similarities)
        if similarities[best_idx] > 0.25:
            return domain_keys[best_idx], float(similarities[best_idx])
            
    # Original Keyword signals fallback
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
