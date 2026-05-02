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
            category = risk_info.get("risk_category")
            
            risk_to_type = {
                "financial_fraud": "product_issue",
                "assessment_integrity": "product_issue", 
                "billing_disputes": "product_issue",
                "security": "bug",
                "legal_privacy": "product_issue",
                "account_access": "product_issue",
                "low_confidence": "product_issue"
            }
            req_type = risk_to_type.get(category, "product_issue")
            
            if category == "low_confidence":
                response_msg = "We couldn't find specific documentation covering your issue. A human agent will review and respond shortly."
            else:
                response_msg = "This ticket has been escalated to a human agent for review due to risk policies."

            output = {
                "status": "escalated",
                "product_area": domain or "unknown",
                "response": response_msg,
                "justification": risk_info["reason"],
                "request_type": req_type
            }
            return output, context

        # 5. Generate Response via LLM
        llm_response = self.responder.generate_response(issue, subject, domain, chunks, risk_info)

        return llm_response, context
