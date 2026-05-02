import os
import random
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from corpus.loader import load_corpus
from retrieval.retriever import HybridRetriever
from pipeline import Pipeline
from utils.csv_io import read_tickets, write_output
from utils.logger import log_ticket_transcript

def main():
    load_dotenv()
    random.seed(42)  # Determinism, per evaluation criteria
    
    parser = argparse.ArgumentParser(description="Run batch triage")
    parser.add_argument("--input", default="../support_tickets/support_tickets.csv")
    parser.add_argument("--output", default="../support_tickets/output.csv")
    parser.add_argument("--data", default="../data/")
    log_dir = os.path.expanduser("~/hackerrank_orchestrate")
    os.makedirs(log_dir, exist_ok=True)
    parser.add_argument("--log", default=os.path.join(log_dir, "log.txt"))
    args = parser.parse_args()
    
    print(f"Loading corpus from {args.data}...")
    corpus_chunks = load_corpus(args.data)
    
    retriever = HybridRetriever(corpus_chunks)
    pipeline = Pipeline(retriever)
    
    print(f"Reading tickets from {args.input}...")
    records = read_tickets(args.input)
    
    results = []
    
    for idx, row in tqdm(enumerate(records), total=len(records), desc="Processing Tickets"):
        issue = row['issue']
        subject = row['subject']
        company = row['company']
        
        try:
            output, context = pipeline.process_ticket(issue, subject, company)
        except Exception as e:
            output = {
                "status": "escalated",
                "product_area": "unknown",
                "response": "An error occurred processing this ticket.",
                "justification": f"Processing error: {str(e)}",
                "request_type": "invalid"
            }
            context = {"domain": "unknown", "chunks": [], "risk_info": {}}
        
        # Only the 5 evaluated columns go into output.csv
        full_row = {
            "status": output.get("status"),
            "product_area": output.get("product_area"),
            "response": output.get("response"),
            "justification": output.get("justification"),
            "request_type": output.get("request_type")
        }
        results.append(full_row)
        
        log_ticket_transcript(
            args.log, 
            ticket_id=idx+1, 
            issue=issue, 
            subject=subject, 
            company=company, 
            domain=context["domain"], 
            chunks=context["chunks"], 
            risk_info=context["risk_info"], 
            output=output
        )
        
    print(f"Writing outputs to {args.output}...")
    write_output(args.output, results)
    print("Done!")

if __name__ == "__main__":
    main()
