from triage.input_validator import validate_input
from triage.domain_detector import detect_domain
from triage.risk_assessor import assess_risk
from triage.responder import Responder

class Pipeline:
    def __init__(self, retriever):
        self.retriever = retriever
        self.responder = Responder()
        
    def process_ticket(self, issue, subject, company):
        """
        Orchestrates the triage logic for a single ticket.
        Returns (output_dict, context_dict)
        """
        context = {
            "domain": None,
            "chunks": [],
            "risk_info": {"risk_level": "none", "should_escalate": False, "reason": ""}
        }
        
        # 1. Validate Input — catch injections and off-domain commands before doing any LLM work
        val_result = validate_input(issue, subject)
        if not val_result["valid"]:
            # Prompt injection = escalate. Off-domain/dangerous command = reply as invalid
            is_injection = "injection" in val_result["reason"].lower()
            output = {
                "status": "escalated" if is_injection else "replied",
                "product_area": "security/invalid",
                "response": (
                    "I cannot process this request."
                    if is_injection
                    else "I'm sorry, this request is out of scope from my capabilities."
                ),
                "justification": val_result["reason"],
                "request_type": val_result["request_type"]
            }
            return output, context
            
        # 2. Detect Domain
        domain, conf = detect_domain(company, f"{subject} {issue}", embedder=self.retriever.embedder)
        context["domain"] = domain
        
        # 3. Retrieve chunks
        query = f"{subject} {issue}"
        chunks, max_score = self.retriever.retrieve(query, domain_filter=domain, top_k=5)
        context["chunks"] = chunks
        
        # 4. Assess Risk
        risk_info = assess_risk(issue, domain, max_score, embedder=self.retriever.embedder)
        context["risk_info"] = risk_info

        # 4b. Escalation gate — short-circuit before touching the LLM
        if risk_info["should_escalate"]:
            # Build a human-readable reason
            reason = risk_info["reason"]
            
            escalation_messages = {
                "financial_fraud": "Your report of unauthorized or fraudulent activity has been flagged for urgent review. Please also contact your bank or card issuer directly for immediate assistance.",
                "account_access": "Your account access issue has been flagged for review. A human agent will follow up to help restore your access.",
                "assessment_integrity": "Your concern regarding assessment integrity has been escalated for review by our team.",
                "billing_disputes": "Your billing dispute has been flagged for human review. A support agent will follow up to assist with your request.",
                "security": "Your security concern has been escalated to our security team for immediate review.",
                "legal_privacy": "Your request involving legal or privacy matters has been escalated to the appropriate team.",
            }
            
            # Match reason to message
            response_message = "Your request has been escalated to a human agent who will follow up shortly."
            for category, message in escalation_messages.items():
                if category in reason:
                    response_message = message
                    break
            
            # Low confidence escalation gets its own message        
            if "No corpus coverage" in reason:
                response_message = (
                    f"We couldn't find specific documentation covering your issue with "
                    f"{domain.capitalize() if domain else 'this product'}. A human agent will review your request "
                    f"and respond shortly."
                )
            
            # Map domain to a clean product_area format using a mapping dictionary
            domain_display = {
                "hackerrank": "HackerRank",
                "claude": "Claude",
                "visa": "Visa"
            }
            product_area = domain_display.get(domain, domain or "Unknown")

            output = {
                "status": "escalated",
                "product_area": product_area,
                "response": response_message,
                "justification": risk_info["reason"],  # just the reason, no category prefix
                "request_type": "product_issue"
            }
            return output, context

        # 5. Generate Response via LLM
        llm_response = self.responder.generate_response(issue, subject, domain, chunks, risk_info)

        return llm_response, context
