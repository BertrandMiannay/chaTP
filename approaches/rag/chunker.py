def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def chunk_documents(
    docs: dict[str, str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    all_chunks = []
    for doc_name, text in docs.items():
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            all_chunks.append({"text": chunk, "doc_name": doc_name, "chunk_index": i})
    return all_chunks
