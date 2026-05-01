# HackerRank Orchestrate: Support Triage Agent

## Overview
This is a terminal-based AI support triage agent that uses **Gemini 2.5 Pro** and a custom **TF-IDF Retrieval Augmented Generation (RAG)** pipeline to classify, route, and answer support tickets for HackerRank, Claude, and Visa.

## Architecture
1. **Retriever (`retriever.py`)**: Efficiently loads the Markdown, HTML, and text corpus from `data/` and builds an in-memory TF-IDF index using `scikit-learn`. For every incoming ticket, it calculates the cosine similarity to find the top 3 most relevant support documents. We used TF-IDF because it is lightning fast, runs 100% locally, requires zero API calls, and avoids rate limits.
2. **Agent (`agent.py`)**: Interfaces with the `google-genai` SDK and utilizes `gemini-2.5-pro` with Structured Outputs (`response_schema`). It reads the top documents, identifies the `request_type`, decides whether to `reply` or `escalate`, and generates a safe, grounded response.
3. **Main Execution (`main.py`)**: Reads the input `support_tickets.csv`, processes them row-by-row, and outputs the final predictions to `output.csv`.

## Prerequisites
- Python 3.9+
- A Gemini API Key

## Setup & Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install pandas python-dotenv google-genai scikit-learn numpy pydantic
   ```

3. Configure Environment Variables:
   Create a `.env` file in the root directory (one level above `code/`) and add:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Running the Agent
Run the main script from the repository root:
```bash
python code/main.py
```

The script will read `support_tickets/support_tickets.csv` and write the populated predictions to `support_tickets/output.csv`.
