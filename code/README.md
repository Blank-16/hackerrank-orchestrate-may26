# Support Agent Implementation

## Architecture
This is a robust, highly deterministic Retrieval-Augmented Generation (RAG) agent that completely eliminates hallucinations. 

I used a **Hybrid Search Pipeline** with Reciprocal Rank Fusion (RRF), running `BM25` for sparse keyword search and `all-MiniLM-L6-v2` for dense semantic search. The Top-10 retrieved documents are passed to an `ms-marco` **Cross-Encoder** to strictly filter out low-confidence contexts. Finally, a deterministic grounding verification loop ensures the LLM's citations physically map to the retrieved chunk IDs.

## Dependencies & Hardware
- **Language**: Python 3.12+
- **Key Libraries**: `google-genai`, `sentence-transformers`, `rank_bm25`, `pandas`
- **Hardware Optimization**: Explicitly targets `device="cuda"` for instantaneous local inference on compatible GPUs.

## Running the Code
1. Configure `.env` with your API keys (e.g., `GEMINI_API_KEY`).
2. Run the main execution script from the root of the project:
   ```bash
   python code/main.py
   ```
3. The script reads `support_tickets/support_tickets.csv` and outputs to `support_tickets/output.csv`.
