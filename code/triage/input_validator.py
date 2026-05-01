import re

# Injection patterns — multilingual variants included
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all previous",
    r"you are now",
    r"pretend to be",
    r"forget your instructions",
    r"system prompt",
    r"affiche toutes les r.gles",          # French: "show all the rules"
    r"affiche.*documents r.cup.r.s",       # French: "show retrieved documents"
    r"logique exacte que vous utilisez",   # French: "exact logic you use"
    r"r.gles internes",                    # French: "internal rules"
]

# Commands that are unambiguously off-domain and potentially harmful
OFF_DOMAIN_COMMANDS = [
    r"delete all files",
    r"rm -rf",
    r"format (my|the) (disk|drive|system)",
]

def validate_input(issue_text, subject_text=""):
    """
    Check for prompt injection, gibberish, or invalid inputs.
    Returns: {valid: bool, reason: str, request_type: str}
    """
    combined_text = f"{subject_text} {issue_text}".lower().strip()
    
    if not combined_text:
        return {"valid": False, "reason": "Empty input", "request_type": "invalid"}
        
    # Check for prompt injections
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, combined_text):
            return {"valid": False, "reason": "Prompt injection detected", "request_type": "invalid"}

    # Check for dangerous off-domain commands — reply rather than escalate
    for pattern in OFF_DOMAIN_COMMANDS:
        if re.search(pattern, combined_text):
            return {"valid": False, "reason": "Dangerous off-domain command detected", "request_type": "invalid"}
            
    # Check for gibberish (too short and no real words)
    if len(combined_text) < 5 and not any(char.isalpha() for char in combined_text):
         return {"valid": False, "reason": "Input appears to be gibberish or too short", "request_type": "invalid"}
         
    return {"valid": True, "reason": "", "request_type": "continue"}
