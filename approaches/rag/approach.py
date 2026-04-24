import json
from pathlib import Path

import faiss
import numpy as np

from api.base import BaseAPI
from approaches.base import BaseApproach, ApproachResponse
from approaches.utils import load_pdf_texts
from approaches.rag.chunker import chunk_documents
from approaches.rag.embedder import MistralEmbedder

SYSTEM_PROMPT = """Tu es un expert en plongée sous-marine.
Réponds uniquement en français, de manière précise et sourcée, en te basant
exclusivement sur les extraits de documents fournis.
Si la réponse n'est pas dans les extraits, dis-le clairement.
Ne fournis jamais de liens externes."""

CACHE_DIR = Path("vector_store")
INDEX_FILE = CACHE_DIR / "index.faiss"
CHUNKS_FILE = CACHE_DIR / "chunks.json"


class RAGApproach(BaseApproach):
    name = "rag"

    def __init__(
        self,
        api: BaseAPI,
        pdf_dir: str = "data",
        top_k: int = 5,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        self.api = api
        self.top_k = top_k
        self.embedder = MistralEmbedder()
        self.chunks, self.index = self._load_or_build_index(pdf_dir, chunk_size, overlap)

    def _load_or_build_index(
        self, pdf_dir: str, chunk_size: int, overlap: int
    ) -> tuple[list[dict], faiss.Index]:
        if INDEX_FILE.exists() and CHUNKS_FILE.exists():
            print("Chargement de l'index FAISS depuis le cache...")
            index = faiss.read_index(str(INDEX_FILE))
            with open(CHUNKS_FILE, encoding="utf-8") as f:
                chunks = json.load(f)
            return chunks, index

        print("Construction de l'index FAISS (premier lancement)...")
        docs = load_pdf_texts(pdf_dir)
        chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.embed_texts(texts)

        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)

        CACHE_DIR.mkdir(exist_ok=True)
        faiss.write_index(index, str(INDEX_FILE))
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Index construit : {len(chunks)} chunks, dimension {dim}")
        return chunks, index

    def _retrieve(self, question: str) -> tuple[list[dict], list[float]]:
        q_vec = self.embedder.embed_query(question).reshape(1, -1)
        distances, indices = self.index.search(q_vec, self.top_k)
        retrieved = []
        scores = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                retrieved.append(self.chunks[idx])
                scores.append(float(dist))
        return retrieved, scores

    def ask(self, question: str) -> ApproachResponse:
        retrieved, scores = self._retrieve(question)

        context_parts = [
            f"=== {c['doc_name']} (extrait {c['chunk_index']}) ===\n{c['text']}"
            for c in retrieved
        ]
        context = "\n\n".join(context_parts)

        messages = [{"role": "user", "content": f"{context}\n\n---\n\nQuestion : {question}"}]
        response = self.api.send(messages, system=SYSTEM_PROMPT)

        metadata = {
            "retrieved_chunks": [
                {
                    "doc_name": c["doc_name"],
                    "chunk_index": c["chunk_index"],
                    "score": round(score, 4),
                    "text_preview": c["text"][:120] + "...",
                }
                for c, score in zip(retrieved, scores)
            ]
        }

        return ApproachResponse(
            answer=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            metadata=metadata,
        )
