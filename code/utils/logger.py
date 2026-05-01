import datetime
import os

def log_ticket_transcript(log_path, ticket_id, issue, subject, company, domain, chunks, risk_info, output):
    """
    Logs the detailed per-ticket processing steps to the main log file.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    retrieval_log = ""
    for c in chunks:
        short_text = c['text'][:70].replace('\n', ' ') + "..."
        retrieval_log += f"  [{c.get('score', 0):.2f}] {c['source']} > \"{short_text}\"\n"
        
    if not retrieval_log:
        retrieval_log = "  [No relevant chunks found]\n"

    lines = [
        f"=== TICKET #{ticket_id} | {timestamp} ===",
        "INPUT:",
        f"  company: {company}",
        f"  subject: {subject}",
        f"  issue: {issue}",
        "",
        f"DOMAIN DETECTED: {domain}",
        "",
        "RETRIEVAL:",
        retrieval_log,
        f"RISK ASSESSMENT: {risk_info.get('risk_level', 'unknown').upper()} | escalate={risk_info.get('should_escalate', False)}",
        "",
        "OUTPUT:",
        f"  status: {output.get('status')}",
        f"  product_area: {output.get('product_area')}",
        f"  request_type: {output.get('request_type')}",
        f"  response: {output.get('response')}",
        f"  justification: {output.get('justification')}",
        "==============================================",
        "",
    ]
    transcript = "\n".join(lines)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(transcript)
