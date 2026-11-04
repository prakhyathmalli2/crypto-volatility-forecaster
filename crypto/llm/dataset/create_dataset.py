from pathlib import Path
import ast
import json
import re


# ============================================================
# PATHS
# ============================================================

# create_dataset.py:
#
# crypto/
#   llm/
#     dataset/
#       create_dataset.py
#
# Therefore parents[2] = crypto/

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    Path(__file__).resolve().parent / "train.jsonl"
)


# ============================================================
# FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".txt",
}


# ============================================================
# DIRECTORIES TO IGNORE
# ============================================================

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    ".venv-llm",
    "__pycache__",
    "saved_models",
    "raw",
    "processed",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are the AI assistant for the Crypto Volatility "
    "Forecast and Smart Signal Generator project. "
    "Answer questions using only the supplied project context. "
    "Do not invent files, functions, models, metrics, or behavior."
)


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_files():

    files = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        relative_parts = path.relative_to(
            PROJECT_ROOT
        ).parts

        if any(
            part in IGNORED_DIRECTORIES
            for part in relative_parts
        ):
            continue

        # Ignore generated RAG database files.
        if "llm" in relative_parts:

            if path.name in {
                "metadata.json",
                "project.index",
            }:
                continue

        files.append(path)

    return sorted(files)


# ============================================================
# RELATIVE FILE NAME
# ============================================================

def relative_name(path):

    return str(
        path.relative_to(PROJECT_ROOT)
    ).replace("\\", "/")


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace("\x00", "")

    text = re.sub(
        r"\n{4,}",
        "\n\n\n",
        text
    )

    return text.strip()


# ============================================================
# ADD EXAMPLE
# ============================================================

def add_example(
    dataset,
    question,
    answer
):

    question = clean_text(question)
    answer = clean_text(answer)

    if not question or not answer:
        return

    dataset.append(
        {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        }
    )


# ============================================================
# PYTHON ANALYSIS
# ============================================================

def analyse_python_file(path):

    try:
        source = path.read_text(
            encoding="utf-8"
        )
    except Exception:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    module_docstring = ast.get_docstring(tree)

    classes = []
    functions = []
    imports = []
    constants = []

    for node in tree.body:

        # ----------------------------------------------------
        # Classes
        # ----------------------------------------------------

        if isinstance(node, ast.ClassDef):

            classes.append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node)
                }
            )

        # ----------------------------------------------------
        # Functions
        # ----------------------------------------------------

        elif isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef
            )
        ):

            args = []

            for arg in node.args.args:
                args.append(arg.arg)

            functions.append(
                {
                    "name": node.name,
                    "docstring": ast.get_docstring(node),
                    "arguments": args
                }
            )

        # ----------------------------------------------------
        # Imports
        # ----------------------------------------------------

        elif isinstance(node, ast.Import):

            for item in node.names:
                imports.append(item.name)

        elif isinstance(node, ast.ImportFrom):

            if node.module:
                imports.append(node.module)

        # ----------------------------------------------------
        # Constants
        # ----------------------------------------------------

        elif isinstance(node, ast.Assign):

            for target in node.targets:

                if isinstance(
                    target,
                    ast.Name
                ):

                    name = target.id

                    if (
                        name.isupper()
                        and isinstance(
                            node.value,
                            (
                                ast.Constant,
                                ast.List,
                                ast.Tuple,
                                ast.Dict
                            )
                        )
                    ):
                        constants.append(name)

    return {
        "source": source,
        "module_docstring": module_docstring,
        "classes": classes,
        "functions": functions,
        "imports": sorted(set(imports)),
        "constants": sorted(set(constants))
    }


# ============================================================
# EXTRACT ROUTES
# ============================================================

def extract_fastapi_routes(source):

    routes = []

    pattern = re.compile(
        r'@(?:app|router)\.(get|post|put|delete|patch)'
        r'\(\s*["\']([^"\']+)["\']'
    )

    for method, route in pattern.findall(source):

        routes.append(
            f"{method.upper()} {route}"
        )

    return sorted(set(routes))


# ============================================================
# EXTRACT SIGNAL ACTIONS
# ============================================================

def extract_signal_actions(source):

    actions = []

    for action in [
        "BUY",
        "HOLD",
        "SELL",
        "REDUCE"
    ]:

        if re.search(
            rf'["\']{action}["\']',
            source
        ):
            actions.append(action)

    return actions


# ============================================================
# EXTRACT MODEL TERMS
# ============================================================

def extract_model_terms(source):

    terms = []

    checks = [
        "GARCH",
        "LSTM",
        "Transformer",
        "ensemble"
    ]

    for term in checks:

        if term.lower() in source.lower():
            terms.append(term)

    return terms


# ============================================================
# GENERATE PYTHON EXAMPLES
# ============================================================

def generate_python_examples(
    dataset,
    path,
    analysis
):

    file_name = relative_name(path)

    source = analysis["source"]

    classes = analysis["classes"]
    functions = analysis["functions"]
    imports = analysis["imports"]
    constants = analysis["constants"]

    module_docstring = analysis[
        "module_docstring"
    ]

    # ========================================================
    # MODULE DOCSTRING
    # ========================================================

    if module_docstring:

        add_example(
            dataset,
            f"What does {file_name} describe?",
            (
                f"The module docstring of {file_name} "
                f"describes it as:\n\n"
                f"{module_docstring}"
            )
        )

    # ========================================================
    # FILE STRUCTURE
    # ========================================================

    structure_parts = []

    if classes:

        structure_parts.append(
            "Classes: "
            + ", ".join(
                item["name"]
                for item in classes
            )
        )

    else:

        structure_parts.append(
            "Classes: None explicitly defined"
        )

    if functions:

        structure_parts.append(
            "Functions: "
            + ", ".join(
                item["name"]
                for item in functions
            )
        )

    else:

        structure_parts.append(
            "Functions: None explicitly defined"
        )

    add_example(
        dataset,
        f"What classes and functions are defined in {file_name}?",
        (
            f"The source of {file_name} contains:\n\n"
            + "\n".join(structure_parts)
        )
    )

    # ========================================================
    # CLASSES
    # ========================================================

    for item in classes:

        name = item["name"]
        docstring = item["docstring"]

        if docstring:

            answer = (
                f"The class {name} is defined in "
                f"{file_name}.\n\n"
                f"Its documented purpose is:\n"
                f"{docstring}"
            )

        else:

            answer = (
                f"The class {name} is defined in "
                f"{file_name}. The source does not "
                f"provide a class-level docstring."
            )

        add_example(
            dataset,
            f"What class is defined in {file_name}?",
            answer
        )

    # ========================================================
    # FUNCTIONS
    # ========================================================

    for item in functions:

        name = item["name"]
        arguments = item["arguments"]
        docstring = item["docstring"]

        signature = (
            f"{name}("
            + ", ".join(arguments)
            + ")"
        )

        if docstring:

            answer = (
                f"The function {signature} is defined "
                f"in {file_name}.\n\n"
                f"Its documented purpose is:\n"
                f"{docstring}"
            )

        else:

            answer = (
                f"The function {signature} is defined "
                f"in {file_name}. The source does not "
                f"provide a function-level docstring."
            )

        add_example(
            dataset,
            f"What function is defined in {file_name}?",
            answer
        )

    # ========================================================
    # IMPORTS
    # ========================================================

    if imports:

        add_example(
            dataset,
            f"What modules are imported by {file_name}?",
            (
                f"The source of {file_name} imports "
                f"or references these modules:\n\n"
                + "\n".join(
                    f"- {item}"
                    for item in imports
                )
            )
        )

    # ========================================================
    # CONSTANTS
    # ========================================================

    if constants:

        add_example(
            dataset,
            f"What uppercase constants are defined in {file_name}?",
            (
                f"The source of {file_name} defines "
                f"these uppercase constants:\n\n"
                + "\n".join(
                    f"- {item}"
                    for item in constants
                )
            )
        )

    # ========================================================
    # FASTAPI ROUTES
    # ========================================================

    routes = extract_fastapi_routes(source)

    if routes:

        add_example(
            dataset,
            f"What API routes are explicitly defined in {file_name}?",
            (
                f"The source of {file_name} explicitly "
                f"defines these API routes:\n\n"
                + "\n".join(
                    f"- {route}"
                    for route in routes
                )
            )
        )

    # ========================================================
    # SIGNAL ACTIONS
    # ========================================================

    actions = extract_signal_actions(source)

    if actions:

        add_example(
            dataset,
            f"Which trading actions are explicitly present in {file_name}?",
            (
                f"The source of {file_name} explicitly "
                f"contains these trading action values:\n\n"
                + ", ".join(actions)
                + "\n\n"
                "This answer only reports action values "
                "present in this file."
            )
        )

    # ========================================================
    # MODEL TERMS
    # ========================================================

    model_terms = extract_model_terms(source)

    if model_terms:

        add_example(
            dataset,
            f"Which model-related terms are explicitly present in {file_name}?",
            (
                f"The source of {file_name} explicitly "
                f"contains these model-related terms:\n\n"
                + ", ".join(model_terms)
                + "\n\n"
                "The presence of a term in this file does "
                "not imply that every related model is "
                "implemented in this file."
            )
        )


# ============================================================
# MARKDOWN / TEXT EXAMPLES
# ============================================================

def generate_text_examples(
    dataset,
    path
):

    file_name = relative_name(path)

    try:
        text = path.read_text(
            encoding="utf-8"
        )
    except Exception:
        return

    text = clean_text(text)

    if not text:
        return

    # Use headings as useful structural information.
    headings = re.findall(
        r"(?m)^#{1,4}\s+(.+)$",
        text
    )

    if headings:

        add_example(
            dataset,
            f"What sections are present in {file_name}?",
            (
                f"The file {file_name} contains these "
                f"headings or sections:\n\n"
                + "\n".join(
                    f"- {heading.strip()}"
                    for heading in headings
                )
            )
        )

    # Include a bounded project-document excerpt.
    excerpt = text[:2500]

    add_example(
        dataset,
        f"What information is explicitly documented in {file_name}?",
        (
            f"The file {file_name} explicitly documents "
            f"the following information:\n\n"
            f"{excerpt}"
        )
    )


# ============================================================
# REMOVE BAD GENERATED EXAMPLES
# ============================================================

def remove_low_quality_examples(dataset):

    cleaned = []

    for item in dataset:

        question = item["messages"][1]["content"]
        answer = item["messages"][2]["content"]

        # Remove the old weak keyword-style wording.
        if (
            "project concepts" in question.lower()
            or "concepts are explicitly mentioned" in question.lower()
        ):
            continue

        # Remove accidental empty answers.
        if len(answer.strip()) < 20:
            continue

        cleaned.append(item)

    return cleaned


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(dataset):

    unique = []
    seen = set()

    for item in dataset:

        user_text = item[
            "messages"
        ][1]["content"]

        assistant_text = item[
            "messages"
        ][2]["content"]

        key = (
            user_text.strip(),
            assistant_text.strip()
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(item)

    return unique


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset():

    print("=" * 60)
    print("GROUNDED QLoRA DATASET GENERATOR")
    print("=" * 60)

    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )

    files = discover_files()

    print()
    print(
        f"Discovered {len(files)} project files."
    )

    dataset = []

    for path in files:

        file_name = relative_name(path)

        print(
            f"Processing: {file_name}"
        )

        if path.suffix.lower() == ".py":

            analysis = analyse_python_file(
                path
            )

            if analysis is not None:

                generate_python_examples(
                    dataset,
                    path,
                    analysis
                )

        else:

            generate_text_examples(
                dataset,
                path
            )

    # ========================================================
    # CLEAN
    # ========================================================

    dataset = remove_low_quality_examples(
        dataset
    )

    dataset = remove_duplicates(
        dataset
    )

    # ========================================================
    # SAVE
    # ========================================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for item in dataset:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("DATASET CREATED")
    print("=" * 60)

    print()
    print(
        f"Examples: {len(dataset)}"
    )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print()
    print(
        "Weak keyword examples: REMOVED"
    )

    print(
        "Duplicate examples: REMOVED"
    )

    print(
        "Dataset generation completed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_dataset()