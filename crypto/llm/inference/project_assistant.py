from pathlib import Path
import sys
import re

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

RAG_DIR = BASE_DIR / "crypto" / "llm" / "rag"
TRAINING_DIR = BASE_DIR / "crypto" / "llm" / "training"

ADAPTER_PATH = TRAINING_DIR / "qwen3-1.7B-qlora"

MODEL_NAME = "Qwen/Qwen3-1.7B"


# ============================================================
# IMPORT RAG
# ============================================================

sys.path.insert(0, str(RAG_DIR))

from retrieve import retrieve


# ============================================================
# STARTUP
# ============================================================

print("=" * 60)
print("PROJECT AI ASSISTANT")
print("=" * 60)

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
else:
    print("GPU: CPU")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("Loading Qwen3...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

print("Base model loaded.")


# ============================================================
# LOAD QLORA ADAPTER
# ============================================================

print("Loading QLoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    str(ADAPTER_PATH),
)

model.eval()

print("QLoRA adapter loaded.")


# ============================================================
# NORMALIZE FILE PATH
# ============================================================

def normalize_path(path):
    return (
        path.replace("\\", "/")
        .strip()
        .lower()
    )


# ============================================================
# DETECT FILE IN QUESTION
# ============================================================

def detect_file(question):

    patterns = [
        r"[\w./\\-]+\.py",
        r"[\w./\\-]+\.md",
        r"[\w./\\-]+\.yaml",
        r"[\w./\\-]+\.yml",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE,
        )

        if match:
            return normalize_path(
                match.group(0)
            )

    return None


# ============================================================
# EXACT FILE FILTER
# ============================================================

def select_file_results(
    results,
    mentioned_file,
):

    if not mentioned_file:
        return results

    selected = []

    for result in results:

        result_file = normalize_path(
            result["file"]
        )

        if (
            result_file.endswith(
                mentioned_file
            )
            or mentioned_file.endswith(
                result_file
            )
        ):
            selected.append(result)

    return selected


# ============================================================
# QUESTION TYPE
# ============================================================

def question_type(question):

    q = question.lower()

    if any(
        word in q
        for word in [
            "endpoint",
            "endpoints",
            "api route",
            "api routes",
            "routes",
        ]
    ):
        return "endpoint"

    if any(
        word in q
        for word in [
            "method",
            "methods",
        ]
    ):
        return "method"

    if any(
        word in q
        for word in [
            "class",
            "classes",
        ]
    ):
        return "class"

    if any(
        word in q
        for word in [
            "function",
            "functions",
        ]
    ):
        return "function"

    if any(
        phrase in q
        for phrase in [
            "what does",
            "what is the purpose",
            "purpose of",
            "describe",
            "explain the file",
            "explain this file",
        ]
    ):
        return "module"
    if any(
        phrase in q
        for phrase in [
            "how is",
            "how does",
            "how are",
            "how do",
            "how is it built",
            "how is it implemented",
            "how does it work",
            "how does it predict",
            "how does it generate",
        ]
    ):
        return "implementation"
    

    return "general"


# ============================================================
# STRUCTURED ANSWER
# ============================================================

def structured_answer(
    question,
    results,
):

    q_type = question_type(question)

    # --------------------------------------------------------
    # MODULE PURPOSE
    # --------------------------------------------------------

    if q_type == "module":

        module_chunks = [
            result
            for result in results
            if result.get("type") == "module"
        ]

        if module_chunks:

            text = module_chunks[0]["text"]

            match = re.search(
                r"MODULE DOCSTRING:\s*(.*)",
                text,
                re.DOTALL,
            )

            if match:

                description = (
                    match.group(1)
                    .strip()
                )

                return description

        return None

    # --------------------------------------------------------
    # ENDPOINTS
    # --------------------------------------------------------

    if q_type == "endpoint":

        endpoints = []

        for result in results:

            if result.get("type") != "endpoint":
                continue

            name = result.get(
                "name",
                "",
            ).strip()

            if name and name not in endpoints:
                endpoints.append(name)

        if endpoints:

            return (
                "The endpoints defined in the file are:\n\n"
                + "\n".join(
                    f"- {endpoint}"
                    for endpoint in endpoints
                )
            )

        return None

    # --------------------------------------------------------
    # METHODS
    # --------------------------------------------------------

    if q_type == "method":

        methods = []

        for result in results:

            if result.get("type") != "method":
                continue

            name = result.get(
                "name",
                "",
            ).strip()

            if name:

                method_name = name.split(
                    "."
                )[-1]

                if method_name not in methods:
                    methods.append(
                        method_name
                    )

        if methods:

            return (
                "The methods defined in the file are:\n\n"
                + "\n".join(
                    f"- {method}"
                    for method in methods
                )
            )

        return None

    # --------------------------------------------------------
    # CLASSES
    # --------------------------------------------------------

    if q_type == "class":

        classes = []

        for result in results:

            if result.get("type") != "class":
                continue

            name = result.get(
                "name",
                "",
            ).strip()

            if name and name not in classes:
                classes.append(name)

        if classes:

            return (
                "The classes defined in the file are:\n\n"
                + "\n".join(
                    f"- {name}"
                    for name in classes
                )
            )

        return None

    # --------------------------------------------------------
    # FUNCTIONS
    # --------------------------------------------------------

    if q_type == "function":

        functions = []

        for result in results:

            if result.get("type") != "function":
                continue

            name = result.get(
                "name",
                "",
            ).strip()

            if name and name not in functions:
                functions.append(name)

        if functions:

            return (
                "The functions defined in the file are:\n\n"
                + "\n".join(
                    f"- {name}"
                    for name in functions
                )
            )

        return None

    return None


# ============================================================
# LLM ANSWER
# ============================================================

def generate_llm_answer(
    question,
    context,
):

    system_prompt = """
You are a project documentation assistant.

Answer ONLY using the supplied project source context.

RULES:

1. The supplied source is authoritative.

2. Never invent files, classes, methods, functions,
   variables, endpoints, parameters, or functionality.

3. If information is explicitly present in the supplied
   context, use it directly.

4. MODULE DOCSTRING describes the module purpose.

5. CLASS entries describe classes.

6. FUNCTION entries describe functions.

7. METHOD entries describe class methods.

8. FASTAPI ENDPOINT entries describe API endpoints.

9. When a specific file is requested, use only that file.

10. Combine multiple relevant chunks from the same file.

10a.10a. For implementation or "how does it work" questions, explain the implementation using the concrete details present in the supplied source, including important layers, parameters, outputs, and losses. Do not reproduce the source code unless the user explicitly asks for code.

11. Do not attribute functionality from another file.

12. If the information genuinely is not present, say:

This information is not available in the retrieved project context.

Keep answers concise and factual.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                "PROJECT SOURCE:\n\n"
                + context
                + "\n\nQUESTION:\n"
                + question
                + "\n\nINSTRUCTION:\n"
+ (
    "Explain the implementation clearly and concisely. "
    "Use the supplied source as evidence. "
    "Do not reproduce the source code."
)
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    input_length = (
        inputs["input_ids"].shape[-1]
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][
        input_length:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    if "<think>" in answer:
        answer = answer.split(
            "<think>"
        )[-1]

    if "</think>" in answer:
        answer = answer.split(
            "</think>"
        )[-1]

    return answer.strip()


# ============================================================
# MAIN ASSISTANT LOOP
# ============================================================

print("\n" + "=" * 60)
print("ASSISTANT READY")
print("=" * 60)

print(
    "Ask questions about your project."
)

print(
    "Type 'exit' to stop."
)


while True:

    question = input(
        "\nYou: "
    ).strip()

    if not question:
        continue

    if question.lower() in [
        "exit",
        "quit",
    ]:

        print(
            "\nAssistant stopped."
        )

        break

    # --------------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------------

    results = retrieve(
        question,
        top_k=8,
    )

    # --------------------------------------------------------
    # EXACT FILE
    # --------------------------------------------------------

    mentioned_file = detect_file(
        question
    )

    selected_results = select_file_results(
        results,
        mentioned_file,
    )

    if not selected_results:
        selected_results = results[:8]

    # --------------------------------------------------------
    # STRUCTURED CODE ANSWER
    # --------------------------------------------------------

    answer = structured_answer(
        question,
        selected_results,
    )

    if answer is not None:

        print("\nAssistant:")
        print(answer)

        continue

    # --------------------------------------------------------
    # BUILD LLM CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for result in selected_results:

        context_parts.append(
            f"FILE: {result['file']}\n"
            f"TYPE: {result.get('type', '')}\n"
            f"NAME: {result.get('name', '')}\n"
            f"SOURCE:\n{result['text']}"
        )

    context = (
        "\n\n"
        + (
            "\n\n"
            + "-" * 60
            + "\n\n"
        ).join(
            context_parts
        )
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    answer = generate_llm_answer(
        question,
        context,
    )

    print("\nAssistant:")
    print(answer)