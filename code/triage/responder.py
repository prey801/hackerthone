import os
import json
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Structured output schema
class TriageResponse(BaseModel):
    status: str = Field(description="Must be exactly 'replied' or 'escalated'")
    product_area: str = Field(description="Must be a consistent, human-readable category in the format 'Domain / Category' (e.g., 'HackerRank / Coding Challenges', 'Visa / Consumer Fraud'). The Domain MUST be exactly 'HackerRank', 'Claude', or 'Visa'. Do NOT use snake_case, underscores, or file paths.")
    response: str = Field(description="The user-facing response grounded ONLY in the corpus, or a clear escalation message. You MUST reference the source documentation (e.g. filename or source path) if you use information from it.")
    justification: str = Field(description="Concise routing rationale")
    request_type: str = Field(description="Must be one of: 'product_issue', 'feature_request', 'bug', 'invalid'")

SYSTEM_TEMPLATE = """You are a support triage agent for {domain}.
Answer ONLY using the provided support documentation excerpts.
Do NOT use any outside knowledge or invent policies.
If you use information from the provided documentation, you MUST reference the source (e.g., the source filename) in your response.
If the documentation partially covers the issue, reply with what is covered and note any gaps. Only escalate if the documentation has NO relevant information at all, or if a risk trigger fires. Do not escalate simply because the answer is incomplete. If you are providing actionable next steps or troubleshooting info, the status MUST be 'replied'.
If you note any gaps or inform the user that their specific issue is not fully covered, you MUST append this exact sentence to the end of your response: "For further assistance, please contact support@hackerrank.com / help@hackerrank.com / your account manager."

The risk assessor has flagged this ticket as: {risk_level} risk.
Escalation required: {escalate_required}. Reason: {escalate_reason}.
If escalation is required, the status MUST be 'escalated', and you should provide a helpful, polite response acknowledging the specific issue (e.g., "We've identified your request involves [Issue]...") and indicating it has been escalated to a human agent for review. Do not provide a generic "This ticket has been escalated" message.

Strict output values:
- status: exactly 'replied' or 'escalated'
- request_type: exactly 'product_issue', 'feature_request', 'bug', or 'invalid'"""


def _load_api_keys():
    """
    Load API keys from environment.
    Reads GEMINI_API_KEYS (comma-separated) first, then falls back to GEMINI_API_KEY.
    Returns a list of non-empty key strings.
    """
    multi = os.environ.get("GEMINI_API_KEYS", "")
    if multi.strip():
        keys = [k.strip() for k in multi.split(",") if k.strip()]
        if keys:
            return keys
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    if single:
        return [single]
    raise EnvironmentError(
        "No Gemini API key found. Set GEMINI_API_KEYS=key1,key2,... or GEMINI_API_KEY=key in .env"
    )


class Responder:
    # gemini-2.5-flash-lite free tier: 10 RPM, 20 RPD per key
    _REQUEST_DELAY_S = 7   # safe gap between calls (~8 RPM)

    def __init__(self):
        self._keys = _load_api_keys()
        self._key_idx = 0
        self._client = genai.Client(api_key=self._keys[0])
        self.model_name = "gemini-2.5-flash-lite"
        print(f"[Responder] Loaded {len(self._keys)} API key(s). Using key 1/{len(self._keys)}.")

    def _next_key(self):
        """Rotate to the next available API key. Returns False if all keys exhausted."""
        self._key_idx += 1
        if self._key_idx >= len(self._keys):
            return False
        self._client = genai.Client(api_key=self._keys[self._key_idx])
        print(f"\n  [Key rotation] Switched to key {self._key_idx + 1}/{len(self._keys)}")
        return True

    def generate_response(self, issue_text, subject_text, domain, chunks, risk_info):
        """
        Calls Gemini to generate the final structured triage response.
        Rotates through API keys on 429 quota errors. Retries on 503 overload.
        """
        system_instruction = SYSTEM_TEMPLATE.format(
            domain=domain if domain else "multiple products",
            risk_level=risk_info["risk_level"],
            escalate_required="yes" if risk_info["should_escalate"] else "no",
            escalate_reason=risk_info["reason"]
        )

        corpus_context = ""
        for i, chunk in enumerate(chunks):
            corpus_context += f"[Chunk {i+1} - source: {chunk['source']}]\n{chunk['text']}\n\n"

        user_prompt = f"""Support ticket:
Subject: {subject_text}
Issue: {issue_text}

Relevant documentation:
{corpus_context if corpus_context else "No relevant documentation found."}"""

        # Rate-limit: stay safely under 10 RPM
        time.sleep(self._REQUEST_DELAY_S)

        last_error = None
        overload_retries = 0

        while True:
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=TriageResponse,
                        temperature=0.0,
                    )
                )

                result = json.loads(response.text)

                # Safety override: risk_assessor always wins on escalation
                if risk_info["should_escalate"]:
                    result["status"] = "escalated"

                result["status"] = result["status"].lower()
                result["request_type"] = result["request_type"].lower()
                return result

            except Exception as e:
                last_error = e
                err_str = str(e)

                is_quota = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_overload = "503" in err_str or "UNAVAILABLE" in err_str

                if is_quota:
                    # Quota exhausted on this key — try the next one
                    print(f"\n  [Quota exhausted on key {self._key_idx + 1}] Rotating to next key...")
                    if self._next_key():
                        time.sleep(2)  # brief pause before retrying with new key
                        continue      # retry the same ticket with new key
                    else:
                        print("  [Key rotation] All API keys exhausted for today.")
                        break

                elif is_overload and overload_retries < 2:
                    # Temporary 503 — wait and retry same key
                    overload_retries += 1
                    wait = 45 * overload_retries
                    print(f"\n  [503 overload, retry {overload_retries}/2] Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                else:
                    # Non-retryable error or too many overload retries
                    break

        return {
            "status": "escalated",
            "product_area": "unknown",
            "response": "An internal error occurred. Escalating to a human agent.",
            "justification": f"API or parsing error: {str(last_error)}",
            "request_type": "invalid"
        }
