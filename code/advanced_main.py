import os
import pandas as pd
from dotenv import load_dotenv
from advanced_agent import AdvancedSupportAgent

def main():
    load_dotenv()
    
    agent = AdvancedSupportAgent()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tickets_path = os.path.join(base_dir, "support_tickets", "support_tickets.csv")
    
    if not os.path.exists(tickets_path):
        print(f"File not found: {tickets_path}")
        return
        
    df = pd.read_csv(tickets_path)
    print(f"Loaded {len(df)} tickets to process with the advanced pipeline.")
    
    results = []
    
    for idx, row in df.iterrows():
        issue = str(row.get("Issue", "")) if pd.notna(row.get("Issue")) else ""
        subject = str(row.get("Subject", "")) if pd.notna(row.get("Subject")) else ""
        company = str(row.get("Company", "")) if pd.notna(row.get("Company")) else ""
        
        print(f"\nProcessing ticket {idx+1}/{len(df)}: {subject} ({company})")
        
        prediction = agent.process_ticket(issue, subject, company)
        
        result_row = row.copy()
        result_row["Status"] = prediction.get("status", "escalated")
        result_row["Product Area"] = prediction.get("product_area", "unknown")
        result_row["Response"] = prediction.get("response", "Escalated.")
        result_row["Justification"] = prediction.get("justification", "Escalated.")
        result_row["Request Type"] = prediction.get("request_type", "invalid")
        
        results.append(result_row)
        print(f"  -> Decision: {result_row['Status']}")
        
    output_df = pd.DataFrame(results)
    output_path = os.path.join(base_dir, "support_tickets", "advanced_output.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved advanced results to {output_path}")

if __name__ == "__main__":
    main()
