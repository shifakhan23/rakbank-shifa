# PDF Q&A

A RAG-based system that answers questions from a PDF document (https://urldefense.com/v3/__https:/arxiv.org/pdf/2512.02556__;!!FKJHrVI!mYYpERYUnw2DcvHYGDYzPVP-AZaNFlHVmJm1NmxhPKoDmJrxdztHStn_P6yuCzNDstHhI6GHV2ZALMMOFLyJkeai0jlGBQ$). Users can ask questions and get answers derived from the document content.

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), OpenAI API key

### 1. Install dependencies

```bash
uv sync
```

### 2. Add your OpenAI API key

Open `config.yaml` and put your key:

```yaml
openai_key: "sk-your-openai-api-key-here"
```

### 3. Run the app

```bash
uv run streamlit run app.py
```

Opens at `http://localhost:8501`.

## What's included

- `extracted_text.txt` and `extracted_images/` — already extracted from the PDF using `prepare_input_file.py` (uses GPT-4.1-mini vision for image descriptions)
- `chroma_db/` — pre-built vector store, ready to use

You don't need to regenerate these. Just install, configure the key, and run. But if the vectors are not there you can rebuild it using below command:

To rebuild the vector store:

```bash
uv run python embedding.py --rebuild
```

## Rebuilding (optional)

To re-extract text from the PDF:

```bash
uv run python prepare_input_file.py
```

## Files

- `app.py` — Streamlit chat UI with streaming responses
- `embedding.py` — Chunks text, embeds with OpenAI, stores in ChromaDB
- `prepare_input_file.py` — Extracts text and images from the PDF
- `config.yaml` — API key config
- `pyproject.toml` — Dependencies
