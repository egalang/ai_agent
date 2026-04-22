import os
import json
import argparse
import numpy as np
from datetime import datetime

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings
)

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "./indexes"
REGISTRY_PATH = os.path.join(BASE_DIR, "registry.json")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:3b"

embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
llm = Ollama(model=LLM_MODEL, request_timeout=120.0)

Settings.embed_model = embed_model
Settings.llm = llm


# -----------------------------
# INIT REGISTRY
# -----------------------------
def ensure_registry():
    os.makedirs(BASE_DIR, exist_ok=True)
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w") as f:
            json.dump({"indexes": []}, f, indent=2)


def load_registry():
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)


def save_registry(data):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


# -----------------------------
# EMBEDDING UTILITIES
# -----------------------------
def embed(text: str):
    """Convert text → embedding vector using Ollama embed model"""
    return np.array(embed_model.get_text_embedding(text))


def cosine(a, b):
    """Cosine similarity"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -----------------------------
# CREATE INDEX
# -----------------------------
def add_index(name, path):
    ensure_registry()
    registry = load_registry()

    for idx in registry["indexes"]:
        if idx["name"] == name:
            print("❌ Index already exists")
            return

    print("📂 Loading documents...")
    documents = SimpleDirectoryReader(path, recursive=True).load_data()

    print("🧠 Building index...")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True
    )

    index_dir = os.path.join(BASE_DIR, name)

    print("💾 Saving index...")
    index.storage_context.persist(index_dir)

    # 🔥 Create a semantic "index profile embedding"
    profile_text = f"{name} - {path} - sample docs"
    profile_vec = embed(profile_text).tolist()

    registry["indexes"].append({
        "name": name,
        "path": index_dir,
        "source": path,
        "profile_embedding": profile_vec,
        "created_at": datetime.now().isoformat()
    })

    save_registry(registry)

    print(f"✅ Index '{name}' added")


# -----------------------------
# LIST INDEXES
# -----------------------------
def list_indexes():
    registry = load_registry()

    if not registry["indexes"]:
        print("No indexes found")
        return

    print("\n📚 Indexes:\n")
    for i in registry["indexes"]:
        print(f"- {i['name']} ({i['source']})")


# -----------------------------
# LOAD INDEX
# -----------------------------
def load_index(path):
    storage_context = StorageContext.from_defaults(
        persist_dir=path
    )
    return load_index_from_storage(storage_context)


# -----------------------------
# EMBEDDING-BASED ROUTER 🔥
# -----------------------------
def select_best_index(query, registry):
    query_vec = embed(query)

    best_score = -1
    best_index = None

    for idx in registry["indexes"]:
        idx_vec = np.array(idx["profile_embedding"])
        score = cosine(query_vec, idx_vec)

        print(f"🔎 {idx['name']} similarity: {score:.4f}")

        if score > best_score:
            best_score = score
            best_index = idx

    return best_index


# -----------------------------
# ASK (AUTO ROUTED)
# -----------------------------
def ask(query):
    registry = load_registry()

    if not registry["indexes"]:
        print("❌ No indexes available")
        return

    print("🧠 Selecting best index (embedding routing)...")

    best = select_best_index(query, registry)

    print(f"\n🎯 Selected index: {best['name']}")

    index = load_index(best["path"])
    qe = index.as_query_engine()

    print("\n🤖 Thinking...\n")
    response = qe.query(query)

    print("\n💬 Answer:\n")
    print(response)


# -----------------------------
# MANUAL QUERY
# -----------------------------
def query_index(name, query):
    registry = load_registry()

    for idx in registry["indexes"]:
        if idx["name"] == name:
            index = load_index(idx["path"])
            qe = index.as_query_engine()
            print(qe.query(query))
            return

    print("❌ Not found")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="cmd")

    add = sub.add_parser("add")
    add.add_argument("name")
    add.add_argument("path")

    sub.add_parser("list")

    q = sub.add_parser("query")
    q.add_argument("name")
    q.add_argument("text")

    askp = sub.add_parser("ask")
    askp.add_argument("text")

    args = parser.parse_args()

    if args.cmd == "add":
        add_index(args.name, args.path)

    elif args.cmd == "list":
        list_indexes()

    elif args.cmd == "query":
        query_index(args.name, args.text)

    elif args.cmd == "ask":
        ask(args.text)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()