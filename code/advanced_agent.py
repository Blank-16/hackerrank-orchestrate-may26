import os
import json
from google import genai
from google.genai import types
from advanced_retriever import AdvancedRetriever
from pydantic import BaseModel, Field

class AgentOutput(BaseModel):
    status: str = Field(description="Must be 'replied' or 'escalated'")
    product_area: str = Field(description="Most relevant support category or domain area")
    response: str = Field(description="Extractive user-facing answer grounded ONLY in the corpus, or a brief escalation message.")
    justification: str = Field(description="Concise explanation of the routing/answering decision")
    request_type: str = Field(description="Must be 'product_issue', 'feature_request', 'bug', or 'invalid'")
    citations: list[str] = Field(description="Array of chunk_ids exactly as provided in the context that were used to formulate the response. Must not be empty if status is replied.")

class AdvancedSupportAgent:
    def __init__(self):
        self.model_name = "gemini-2.5-pro"
        try:
            self.client = genai.Client()
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
            self.client = None
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        self.retriever = AdvancedRetriever(data_dir)
        
        # Query normalization dict
        self.norm_dict = {
            "login": "authentication",
            "password": "authentication",
            "payment": "billing",
            "charge": "billing",
            "down": "outage"
        }

    def _normalize_query(self, text: str) -> str:
        words = text.split()
        normalized = [self.norm_dict.get(w.lower(), w) for w in words]
        return " ".join(normalized)

    def _is_high_risk(self, text: str) -> bool:
        risk_keywords = ["hacked", "stolen", "fraud", "bug", "feature request", "vulnerability"]
        return any(word in text.lower() for word in risk_keywords)

    def _validate_grounding(self, result: dict, retrieved_chunks: list) -> tuple[bool, str]:
        if result.get("status") == "escalated":
            return True, "valid_escalation"
            
        citations = result.get("citations", [])
        if not citations:
            return False, "missing_citations"
            
        valid_chunk_ids = {chunk.chunk_id for chunk in retrieved_chunks}
        for citation in citations:
            if citation not in valid_chunk_ids:
                return False, f"invalid_citation_{citation}"
                
        return True, "valid_grounding"

    def process_ticket(self, issue: str, subject: str, company: str):
        if not self.client:
            return self._escalation_response("API client not initialized.")
            
        raw_query = f"{issue} {subject}"
        
        # 1. Deterministic Safety Override
        if self._is_high_risk(raw_query):
            return self._escalation_response("High-risk or unsupported topic detected (e.g. fraud, bug).")
            
        # 2. Query Normalization
        norm_query = self._normalize_query(raw_query)
        
        # 3. Determine company if missing
        company_clean = str(company).strip().lower()
        if company_clean not in ["hackerrank", "claude", "visa"]:
            if "hackerrank" in norm_query.lower(): company_clean = "hackerrank"
            elif "claude" in norm_query.lower() or "anthropic" in norm_query.lower(): company_clean = "claude"
            elif "visa" in norm_query.lower(): company_clean = "visa"
            else: company_clean = "unknown"
            
        # 4. Retrieval & Confidence Gating
        docs, gate_status = self.retriever.retrieve(norm_query, company_clean, top_k=10)
        
        if gate_status != "pass":
            return self._escalation_response(f"Escalated due to retrieval gate: {gate_status}")
            
        # 5. Build Context
        context_text = ""
        for doc in docs:
            context_text += f"\n--- [CHUNK_ID: {doc.chunk_id}] ---\n{doc.text}\n"

        base_prompt = f"""
        You are a customer support triage agent for {company_clean if company_clean != 'unknown' else 'HackerRank, Claude, and Visa'}.
        
        # RULES
        1. Answer the user's issue using ONLY the provided context chunks.
        2. You must output strictly extractive answers. Do NOT hallucinate or add outside facts.
        3. You MUST cite the exact [CHUNK_ID] used in the `citations` array.
        4. If the issue cannot be answered with the context, you MUST return status="escalated".
        
        # TICKET DETAILS
        Query: {norm_query}
        
        # CONTEXT
        {context_text}
        """
        
        strict_retry_prompt = f"""
        PREVIOUS ATTEMPT FAILED VALIDATION (Hallucination or Schema Error).
        CRITICAL: You MUST extract the answer verbatim from the provided context chunks. 
        If the exact answer is not physically present in the text, you MUST return status="escalated".
        Do not add any external information. Cite exact chunk_ids.
        
        # TICKET DETAILS
        Query: {norm_query}
        
        # CONTEXT
        {context_text}
        """

        # Attempt 1
        result, error_msg = self._call_llm(base_prompt)
        if result and self._validate_grounding(result, docs)[0]:
            return result
            
        # Attempt 2 (Retry)
        retry_result, retry_error = self._call_llm(strict_retry_prompt)
        if retry_result and self._validate_grounding(retry_result, docs)[0]:
            return retry_result
            
        # Fallback to escalation
        return self._escalation_response(f"Escalated due to validation failure after retry. Last error: {retry_error}")

    def _call_llm(self, prompt: str) -> tuple[dict, str]:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AgentOutput,
                    temperature=0.0
                )
            )
            result = json.loads(response.text)
            
            # Enum safety check
            if result.get("status") not in ["replied", "escalated"]:
                result["status"] = "escalated"
            if result.get("request_type") not in ["product_issue", "feature_request", "bug", "invalid"]:
                result["request_type"] = "invalid"
                
            return result, "success"
        except Exception as e:
            return None, str(e)

    def _escalation_response(self, justification: str) -> dict:
        return {
            "status": "escalated",
            "product_area": "unknown",
            "response": "I am escalating this ticket to a human agent.",
            "justification": justification,
            "request_type": "invalid",
            "citations": []
        }
