import os
from dotenv import load_dotenv

from corpus.loader import load_corpus
from retrieval.retriever import HybridRetriever
from pipeline import Pipeline

def main():
    load_dotenv()
    
    print("Initializing Domain Support Triage Agent...")
    print("Loading corpus and building retrieval indexes... this may take a moment.")
    
    data_dir = "../data/"
    corpus_chunks = load_corpus(data_dir)
    retriever = HybridRetriever(corpus_chunks)
    pipeline = Pipeline(retriever)
    
    print("=========================================")
    print(" Domain Support Triage Agent Ready")
    print("=========================================")
    
    while True:
        try:
            print("\n--- New Ticket ---")
            company = input("Company [HackerRank/Claude/Visa/None]: ").strip()
            if not company or company.lower() == 'none':
                company = None
                
            subject = input("Subject: ").strip()
            issue = input("Issue: ").strip()
            
            if not issue and not subject:
                print("Empty ticket. Exiting.")
                break
                
            print("\n[Processing...]")
            output, context = pipeline.process_ticket(issue, subject, company)
            
            print(f"\n[Retrieving relevant documentation... Found {len(context['chunks'])} chunks for domain '{context['domain']}']")
            risk_level = context['risk_info'].get('risk_level', 'unknown').upper()
            escalate_str = "ESCALATING" if context['risk_info'].get('should_escalate') else "proceeding to reply"
            print(f"[Risk assessment: {risk_level} — {escalate_str}]")
            
            print("\nOUTPUT:")
            print(f"STATUS:       {output.get('status')}")
            print(f"PRODUCT AREA: {output.get('product_area')}")
            print(f"REQUEST TYPE: {output.get('request_type')}")
            print(f"RESPONSE:     {output.get('response')}")
            print(f"JUSTIFICATION: {output.get('justification')}")
            print("-----------------------------------------")
            
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
