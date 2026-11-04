from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


RAG_DIR = Path(__file__).resolve().parent

VECTOR_FILE = RAG_DIR / "project.index"
METADATA_FILE = RAG_DIR / "metadata.json"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

print("Loading RAG database...")

index = faiss.read_index(str(VECTOR_FILE))

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer(EMBEDDING_MODEL)

print(f"Loaded {index.ntotal} vectors.")


def normalize_path(path):
    return (
        path.replace("\\", "/")
        .strip()
        .lower()
    )


def find_exact_file(query):
    """
    Return all chunks belonging to a file explicitly
    mentioned in the question.
    """

    query_normalized = normalize_path(query)

    matched_files = []

    for item in metadata:

        file_path = normalize_path(
            item["file"]
        )

        filename = Path(file_path).name

        if file_path in query_normalized:
            matched_files.append(
                file_path
            )

        elif query_normalized.endswith(
            file_path
        ):
            matched_files.append(
                file_path
            )

        elif filename in query_normalized:
            matched_files.append(
                file_path
            )

    if not matched_files:
        return []

    target_file = matched_files[0]

    return [
        item
        for item in metadata
        if normalize_path(item["file"])
        == target_file
    ]


def chunk_relevance(item, query):
    """
    Give a small boost to chunk types that are directly
    relevant to the user's question.
    """

    q = query.lower()
    chunk_type = item.get("type", "").lower()
    name = item.get("name", "").lower()

    score = 0.0

    # Endpoint questions
    if any(
        word in q
        for word in [
            "endpoint",
            "api",
            "route",
            "routes",
        ]
    ):
        if chunk_type == "endpoint":
            score += 10

    # Method questions
    if any(
        word in q
        for word in [
            "method",
            "methods",
        ]
    ):
        if chunk_type == "method":
            score += 10

        if chunk_type == "class":
            score += 5

    # Function questions
    if any(
        word in q
        for word in [
            "function",
            "functions",
        ]
    ):
        if chunk_type == "function":
            score += 10

    # Class questions
    if any(
        word in q
        for word in [
            "class",
            "classes",
        ]
    ):
        if chunk_type == "class":
            score += 10

    # General "what does it do?" questions
    if any(
        phrase in q
        for phrase in [
            "what does",
            "what is",
            "purpose",
            "explain",
            "describe",
        ]
    ):
        if chunk_type == "module":
            score += 5

        if chunk_type in {
            "class",
            "function",
            "method",
        }:
            score += 2

    # Name match
    if name and name in q:
        score += 3

    return score
def implementation_relevance(item, query):
    """
    Extra ranking for implementation / how-it-works questions.
    Designed for code-aware RAG.
    """

    q = query.lower()
    text = item.get("text", "").lower()
    name = item.get("name", "").lower()
    chunk_type = item.get("type", "").lower()

    score = 0.0

    implementation_phrases = [
        "how is",
        "how does",
        "how are",
        "how do",
        "how is it built",
        "how is it implemented",
        "how does it work",
        "how does it generate",
        "how does it predict",
        "architecture",
        "implementation",
        "built",
        "implemented",
        "generate",
        "predict",
    ]

    is_implementation = any(
        phrase in q
        for phrase in implementation_phrases
    )

    if not is_implementation:
        return score

    # Prefer actual implementation chunks.
    if chunk_type == "function":
        score += 8

    elif chunk_type == "method":
        score += 6

    elif chunk_type == "class":
        score += 5

    elif chunk_type == "module":
        score += 1

    # Strong boost for implementation-related names.
    implementation_names = [
        "build",
        "create",
        "construct",
        "predict",
        "forward",
        "encode",
        "train",
        "fit",
    ]

    for word in implementation_names:
        if word in name:
            score += 4

    # Strong source-level signals.
    if "def build_" in text:
        score += 6

    if "def predict" in text:
        score += 5

    if "multiheadattention" in text:
        score += 3

    if "globalaveragepooling1d" in text:
        score += 3

    if "volatility" in text:
        score += 2

    if "direction" in text:
        score += 2

    return score

def retrieve(query, top_k=5):

    # --------------------------------------------------
    # 1. Check for exact file
    # --------------------------------------------------

    exact_chunks = find_exact_file(query)

    if exact_chunks:

        # --------------------------------------------------
        # General file questions should always receive the
        # module-level information first.
        # --------------------------------------------------

        general_question = any(
            phrase in query.lower()
            for phrase in [
                "what does",
                "what is",
                "purpose",
                "explain",
                "describe",
            ]
        )

        if general_question:

            module_chunks = [
                item
                for item in exact_chunks
                if item.get("type") == "module"
            ]

            other_chunks = [
                item
                for item in exact_chunks
                if item.get("type") != "module"
            ]

            ranked_other = sorted(
                other_chunks,
                key=lambda item: chunk_relevance(
                    item,
                    query,
                ),
                reverse=True,
            )

            ranked_chunks = module_chunks + ranked_other

        else:

            ranked_chunks = sorted(
                exact_chunks,
                key=lambda item: chunk_relevance(
                    item,
                    query,
                ),
                reverse=True,
            )

        results = []

        for item in ranked_chunks[:top_k]:

            results.append(
                {
                    "file": item["file"],
                    "type": item.get(
                        "type",
                        "",
                    ),
                    "name": item.get(
                        "name",
                        "",
                    ),
                    "text": item["text"],
                    "score": 1.0,
                }
            )

        return results

   # --------------------------------------------------
    # 2. Semantic retrieval
    # --------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Retrieve more candidates than we finally return.
    candidate_k = min(
        max(top_k * 4, 20),
        index.ntotal,
    )

    distances, indices = index.search(
        query_embedding,
        candidate_k,
    )

    candidates = []

    for distance, idx in zip(
        distances[0],
        indices[0],
    ):

        if idx < 0:
            continue

        item = metadata[idx]

        semantic_score = float(distance)

        code_score = chunk_relevance(
            item,
            query,
        )

        implementation_score = implementation_relevance(
            item,
            query,
        )

        final_score = (
            semantic_score
            + code_score
            + implementation_score
        )

        candidates.append(
            {
                "file": item["file"],
                "type": item.get(
                    "type",
                    "",
                ),
                "name": item.get(
                    "name",
                    "",
                ),
                "text": item["text"],
                "score": final_score,
                "_semantic_score": semantic_score,
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    results = []

    for item in candidates[:top_k]:

        results.append(
            {
                "file": item["file"],
                "type": item["type"],
                "name": item["name"],
                "text": item["text"],
                "score": item["score"],
            }
        )

    return results


if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:

        print(
            "Usage:\n"
            'python retrieve.py "your question"'
        )

        sys.exit(1)

    query = " ".join(
        sys.argv[1:]
    )

    results = retrieve(query)

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for i, result in enumerate(
        results,
        1,
    ):

        print(
            f"\nResult {i}"
        )

        print(
            "-" * 60
        )

        print(
            f"FILE: {result['file']}"
        )

        print(
            f"TYPE: {result['type']}"
        )

        print(
            f"NAME: {result['name']}"
        )

        print(
            f"SCORE: {result['score']:.4f}"
        )

        print("\nSOURCE:")

        print(
            result["text"][:2000]
        )