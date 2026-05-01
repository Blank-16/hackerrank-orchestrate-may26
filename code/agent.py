import os
import json
from google import genai
from google.genai import types
from retriever import DocumentRetriever
from pydantic import BaseModel, Field

class AgentOutput(BaseModel):
    status: str = Field(description="Must be 'replied' or 'escalated'")
    product_area: str = Field(description="Most relevant support category or domain area")
    response: str = Field(description="User-facing answer grounded in the corpus, or a brief escalation message")
    justification: str = Field(description="Concise explanation of the routing/answering decision")
    request_type: str = Field(description="Must be 'product_issue', 'feature_request', 'bug', or 'invalid'")

class SupportAgent:
    def __init__(self):
        # We will use gemini-2.5-pro for best results
        self.model_name = "gemini-2.5-pro"
        try:
            self.client = genai.Client()
        except Exception as e:
            print(f"Failed to initialize Gemini client. Is GEMINI_API_KEY set? Error: {e}")
            self.client = None
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        self.retriever = DocumentRetriever(data_dir)

    def process_ticket(self, issue: str, subject: str, company: str):
        if not self.client:
            return {"status": "escalated", "response": "API client not initialized."}
            
        # Determine company if missing
        company_clean = str(company).strip().lower()
        if company_clean not in ["hackerrank", "claude", "visa"]:
            text_to_check = (issue + " " + subject).lower()
            if "hackerrank" in text_to_check:
                company_clean = "hackerrank"
            elif "claude" in text_to_check or "anthropic" in text_to_check:
                company_clean = "claude"
            elif "visa" in text_to_check:
                company_clean = "visa"
            else:
                company_clean = "unknown"
                
        # Retrieve context
        docs = self.retriever.retrieve(issue + " " + subject, company_clean, top_k=3)
        
        context_text = ""
        for i, doc in enumerate(docs):
            context_text += f"\n--- Document {i+1} ---\n{doc['content']}\n"
            
        if not context_text.strip():
            context_text = "No relevant support articles found in the corpus."

        prompt = f"""
You are a customer support triage agent for HackerRank, Claude, and Visa.
Your task is to analyze the support ticket and generate a structured JSON response.

# RULES
1. You MUST use ONLY the provided "Support Context" to answer the issue. Do NOT hallucinate policies or use outside knowledge.
2. If the issue is a bug, a feature request, involves sensitive situations (fraud, account takeover, private info deletion where tools are missing), or is out-of-scope/unsupported, you MUST escalate it.
3. If you escalate, provide a brief empathetic response to the user acknowledging the escalation.
4. Output must match the requested schema exactly.

# TICKET DETAILS
Subject: {subject}
Issue: {issue}
Inferred Company: {company_clean}

# SUPPORT CONTEXT
{context_text}
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AgentOutput,
                    temperature=0.1
                )
            )
            
            result_text = response.text
            result = json.loads(result_text)
            
            # Post-process to ensure strict enum values just in case
            if result.get("status") not in ["replied", "escalated"]:
                result["status"] = "escalated"
            if result.get("request_type") not in ["product_issue", "feature_request", "bug", "invalid"]:
                result["request_type"] = "invalid"
                
            return result
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return {
                "status": "escalated",
                "product_area": "system_error",
                "response": "I apologize, but I am unable to process your request at this moment and will escalate this to a human agent.",
                "justification": f"System error during LLM generation: {str(e)}",
                "request_type": "invalid"
            }
