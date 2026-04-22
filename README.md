# Local Multi-Index RAG Router (Ollama + LlamaIndex)

A local Retrieval-Augmented Generation (RAG) CLI that lets you:

* Build multiple document indexes
* Route questions automatically to the best index using embedding similarity
* Query a specific index manually
* Run everything locally using **Ollama** and **LlamaIndex**

Uses:

* Ollama for embeddings + LLM inference
* LlamaIndex for indexing and retrieval

---

## Features

* Fully local (no cloud APIs required)
* Multi-index registry (`registry.json`)
* Semantic routing using cosine similarity
* Manual and auto-routed query modes
* Persistent vector indexes on disk

---

## Project Structure

```text
project/
├── rag.py
├── indexes/
│   ├── registry.json
│   ├── school_docs/
│   └── finance_docs/
└── documents/
```

---

## Requirements

### System Requirements

Recommended:

* Python 3.10+
* 8GB+ RAM (16GB preferred)
* SSD storage recommended
* Local installation of Ollama

---

## Install Ollama

### Linux / macOS

Install:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
```

Start Ollama:

```bash
ollama serve
```

---

### Windows

1. Download installer:

https://ollama.com/download

2. Install and launch Ollama

3. Verify:

```powershell
ollama --version
```

4. Start service (if needed):

```powershell
ollama serve
```

---

## Pull Required Models

Embedding model:

```bash
ollama pull nomic-embed-text
```

LLM model:

```bash
ollama pull qwen2.5-coder:3b
```

Optional alternatives:

```bash
ollama pull mistral
ollama pull llama3
ollama pull deepseek-coder
```

---

## Create Python Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install --upgrade pip
pip install llama-index
pip install llama-index-embeddings-ollama
pip install llama-index-llms-ollama
pip install numpy
```

Or use a requirements file:

### requirements.txt

```txt
llama-index
llama-index-embeddings-ollama
llama-index-llms-ollama
numpy
```

Install:

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit:

```python
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:3b"
```

Change models if desired.

---

# Usage

## Add an Index

Create an index from a folder of documents:

```bash
python rag.py add school_docs ./documents/schools
```

Example:

```bash
python rag.py add finance_docs ./documents/finance
```

---

## List Indexes

```bash
python rag.py list
```

Example output:

```text
- school_docs
- finance_docs
```

---

## Query a Specific Index

```bash
python rag.py query school_docs "What is tuition?"
```

---

## Auto-Routed Query

Automatically selects the best index:

```bash
python rag.py ask "How do I process payroll taxes?"
```

---

## How Routing Works

Query:

```text
How do I process payroll taxes?
```

The system:

1. Converts query into embedding

2. Compares similarity against index profile embeddings

3. Selects best matching index

4. Loads that index

5. Runs retrieval + generation

---

## Supported Documents

Via LlamaIndex readers:

* TXT
* Markdown
* PDF
* DOCX
* CSV
* Others supported by LlamaIndex

---

## Troubleshooting

## Ollama not running

Error:

```text
Connection refused
```

Fix:

```bash
ollama serve
```

---

## Model missing

Error:

```text
model not found
```

Fix:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:3b
```

---

## Empty or poor answers

Possible causes:

* Weak source documents
* Wrong model selected
* Weak index profile embeddings

Improve routing by replacing:

```python
profile_text = f"{name} - {path} - sample docs"
```

with a content summary from documents.

---

## Example Workflow

```bash
# Start Ollama
ollama serve

# Add indexes
python rag.py add school_docs ./documents/schools
python rag.py add finance_docs ./documents/finance

# List indexes
python rag.py list

# Ask routed question
python rag.py ask "How do I compute taxes?"
```

---

## Development Ideas

Future improvements:

* FastAPI service
* Docker Compose deployment
* Web chat frontend
* Multi-user tenancy
* Top-K routing (query multiple indexes)
* Metadata filtering

---

## License

MIT

---

## Acknowledgments

Powered by:

* Ollama
* LlamaIndex
