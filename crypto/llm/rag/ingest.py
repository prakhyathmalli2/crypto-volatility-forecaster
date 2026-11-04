from pathlib import Path
import ast
import json
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[3]

RAG_DIR = BASE_DIR / "crypto" / "llm" / "rag"

VECTOR_FILE = RAG_DIR / "project.index"
METADATA_FILE = RAG_DIR / "metadata.json"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# --------------------------------------------------
# Files to index
# --------------------------------------------------

EXTENSIONS = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".txt",
}

IGNORE_DIRS = {
    ".git",
    ".venv",
    ".venv-llm",
    "__pycache__",
    "saved_models",
    "raw",
    "processed",
}

IGNORE_FILES = {
    "metadata.json",
    "project.index",
}


# --------------------------------------------------
# Python AST extraction
# --------------------------------------------------

def extract_python_chunks(path, source):
    """
    Extract focused, code-aware chunks from Python files.
    """

    chunks = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunks

    # --------------------------------------------------
    # Module documentation
    # --------------------------------------------------

    module_doc = ast.get_docstring(tree)

    if module_doc:
        chunks.append(
            {
                "type": "module",
                "name": "module",
                "text": (
                    f"FILE: {path}\n"
                    f"MODULE DOCSTRING:\n{module_doc}"
                ),
            }
        )

    # --------------------------------------------------
    # Imports
    # --------------------------------------------------

    imports = []

    for node in tree.body:

        if isinstance(node, ast.Import):

            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):

            module = node.module or ""

            for alias in node.names:
                imports.append(
                    f"{module}.{alias.name}"
                )

    if imports:
        chunks.append(
            {
                "type": "imports",
                "name": "imports",
                "text": (
                    f"FILE: {path}\n"
                    f"IMPORTS:\n"
                    + "\n".join(
                        f"- {item}"
                        for item in imports
                    )
                ),
            }
        )

    # --------------------------------------------------
    # Classes
    # --------------------------------------------------

    for node in ast.walk(tree):

        if isinstance(node, ast.ClassDef):

            methods = []

            for child in node.body:

                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    methods.append(
                        child.name
                    )

            doc = ast.get_docstring(node)

            text = (
                f"FILE: {path}\n"
                f"CLASS: {node.name}\n"
            )

            if doc:
                text += (
                    f"DESCRIPTION:\n{doc}\n"
                )

            if methods:
                text += (
                    "METHODS:\n"
                    + "\n".join(
                        f"- {method}"
                        for method in methods
                    )
                )

            chunks.append(
                {
                    "type": "class",
                    "name": node.name,
                    "text": text,
                }
            )

            # --------------------------------------------------
            # Individual methods
            # --------------------------------------------------

            for child in node.body:

                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):

                    try:
                        method_source = ast.get_source_segment(
                            source,
                            child,
                        )
                    except Exception:
                        method_source = None

                    if not method_source:
                        continue

                    chunks.append(
                        {
                            "type": "method",
                            "name": (
                                f"{node.name}.{child.name}"
                            ),
                            "text": (
                                f"FILE: {path}\n"
                                f"CLASS: {node.name}\n"
                                f"METHOD: {child.name}\n"
                                f"SOURCE CODE:\n"
                                f"{method_source}"
                            ),
                        }
                    )

    # --------------------------------------------------
    # Top-level functions
    # --------------------------------------------------

    for node in tree.body:

        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):

            try:
                function_source = ast.get_source_segment(
                    source,
                    node,
                )
            except Exception:
                function_source = None

            if not function_source:
                continue

            chunks.append(
                {
                    "type": "function",
                    "name": node.name,
                    "text": (
                        f"FILE: {path}\n"
                        f"FUNCTION: {node.name}\n"
                        f"SOURCE CODE:\n"
                        f"{function_source}"
                    ),
                }
            )

    # --------------------------------------------------
    # FastAPI endpoints
    # --------------------------------------------------

    endpoint_pattern = re.compile(
        r'@app\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)'
    )

    for match in endpoint_pattern.finditer(source):

        method = match.group(1).upper()
        route = match.group(2)

        chunks.append(
            {
                "type": "endpoint",
                "name": f"{method} {route}",
                "text": (
                    f"FILE: {path}\n"
                    f"FASTAPI ENDPOINT:\n"
                    f"{method} {route}"
                ),
            }
        )

    return chunks


# --------------------------------------------------
# Markdown / text extraction
# --------------------------------------------------

def extract_text_chunks(path, source):

    chunks = []

    lines = source.splitlines()

    # Headings
    headings = []

    for line in lines:

        if re.match(r"^\s{0,3}#{1,6}\s+", line):

            heading = re.sub(
                r"^\s{0,3}#{1,6}\s+",
                "",
                line,
            ).strip()

            headings.append(heading)

    for heading in headings:

        chunks.append(
            {
                "type": "heading",
                "name": heading,
                "text": (
                    f"FILE: {path}\n"
                    f"SECTION: {heading}"
                ),
            }
        )

    # Bounded full-text chunks
    chunk_size = 1200

    for start in range(
        0,
        len(lines),
        40,
    ):

        block = "\n".join(
            lines[start:start + 40]
        ).strip()

        if not block:
            continue

        chunks.append(
            {
                "type": "text",
                "name": f"lines_{start + 1}",
                "text": (
                    f"FILE: {path}\n"
                    f"TEXT:\n{block}"
                ),
            }
        )

    return chunks


# --------------------------------------------------
# Discover project files
# --------------------------------------------------

def discover_files():

    files = []

    for path in BASE_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.name in IGNORE_FILES:
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        relative_parts = path.relative_to(
            BASE_DIR
        ).parts

        if any(
            part in IGNORE_DIRS
            for part in relative_parts
        ):
            continue

        files.append(path)

    return files


# --------------------------------------------------
# Main ingestion
# --------------------------------------------------

def main():

    print("=" * 60)
    print("BUILDING CODE-AWARE RAG DATABASE")
    print("=" * 60)

    files = discover_files()

    print(
        f"\nFound {len(files)} project files."
    )

    chunks = []

    for path in files:

        relative_path = path.relative_to(
            BASE_DIR
        ).as_posix()

        try:
            source = path.read_text(
                encoding="utf-8"
            )
        except Exception as exc:

            print(
                f"Skipping {relative_path}: {exc}"
            )
            continue

        if path.suffix.lower() == ".py":

            file_chunks = extract_python_chunks(
                relative_path,
                source,
            )

        else:

            file_chunks = extract_text_chunks(
                relative_path,
                source,
            )

        chunks.extend(file_chunks)

        print(
            f"{relative_path}: "
            f"{len(file_chunks)} chunks"
        )

    if not chunks:
        raise RuntimeError(
            "No chunks were generated."
        )

    print(
        f"\nTotal chunks: {len(chunks)}"
    )

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        "Generating embeddings..."
    )

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    RAG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(VECTOR_FILE),
    )

    metadata = []

    for chunk in chunks:

        metadata.append(
            {
                "file": chunk["text"].split(
                    "\n",
                    1,
                )[0].replace(
                    "FILE: ",
                    "",
                ),
                "type": chunk["type"],
                "name": chunk["name"],
                "text": chunk["text"],
            }
        )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("RAG BUILD COMPLETE")
    print("=" * 60)

    print(
        f"Files indexed : {len(files)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print(
        f"Embedding dim : {dimension}"
    )

    print(
        f"Vector index  : {VECTOR_FILE}"
    )

    print(
        f"Metadata      : {METADATA_FILE}"
    )


if __name__ == "__main__":
    main()