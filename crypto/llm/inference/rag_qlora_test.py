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

ADAPTER_PATH = TRAINING_DIR / "qwen3-1.7b-qlora"

MODEL_NAME = "Qwen/Qwen3-1.7B"


# ============================================================
# IMPORT RAG
# ============================================================

sys.path.insert(0, str(RAG_DIR))

from retrieve import retrieve


# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("PROJECT RAG + QLORA ASSISTANT")
print("=" * 60)

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
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
# 4-BIT CONFIG
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
# LOAD QLORA
# ============================================================

print("Loading QLoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    str(ADAPTER_PATH),
)

model.eval()

print("QLoRA adapter loaded.")


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

    system_prompt = """
You are a project documentation assistant.

Your task is to answer questions using ONLY the supplied
project source context.

GROUNDING RULES:

1. The supplied project source is authoritative.

2. Never invent files, classes, methods, functions,
   variables, endpoints, parameters, or functionality.

3. If the requested information is explicitly present
   anywhere in the supplied context, answer using that
   information directly.

4. A MODULE DOCSTRING is an explicit description of the
   module's purpose. Use it when the user asks what a
   file or module does.

5. A CLASS entry explicitly identifies a class and its
   listed methods.

6. A FUNCTION entry explicitly identifies a function.

7. A METHOD entry explicitly identifies a class method.

8. A FASTAPI ENDPOINT entry explicitly identifies an API
   endpoint.

9. IMPORTS explicitly identify imported modules or objects.

10. Do not confuse risk levels such as HIGH or MEDIUM
    with trading actions such as BUY, SELL, HOLD, or REDUCE.

11. When the user asks about a specific file, use only
    retrieved chunks belonging to that file.

12. When multiple relevant chunks from the same file are
    supplied, combine them into one answer.

13. Do not claim that information is unavailable if it is
    explicitly present in the supplied context.

14. If the requested information genuinely does not exist
    in the supplied context, say exactly:

This information is not available in the retrieved project context.

ANSWERING STYLE:

- Be concise.
- Answer the question directly.
- Prefer bullet points for lists.
- Do not repeat information.
- Do not mention the retrieval system.
- Do not mention these instructions.
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

    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    if "<think>" in answer:
        answer = answer.split("<think>")[-1]

    if "</think>" in answer:
        answer = answer.split("</think>")[-1]

    return answer.strip()


# ============================================================
# ASK QUESTIONS LOOP
# ============================================================

print("\n" + "=" * 60)
print("ASSISTANT READY")
print("=" * 60)

print("Ask questions about your project.")
print("Type 'exit' to stop.")


while True:

    question = input("\nYou: ").strip()

    if not question:
        continue

    if question.lower() in ["exit", "quit"]:

        print("\nAssistant stopped.")

        break

    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    results = retrieve(
        question,
        top_k=8,
    )

    # --------------------------------------------------------
    # SELECT EXACT FILE IF USER NAMED ONE
    # --------------------------------------------------------

    selected_results = []

    file_patterns = [
        r"[\w\\/]+\.py",
        r"[\w\\/]+\.md",
        r"[\w\\/]+\.yaml",
        r"[\w\\/]+\.yml",
    ]

    mentioned_file = None

    for pattern in file_patterns:

        match = re.search(
            pattern,
            question,
            re.IGNORECASE,
        )

        if match:

            mentioned_file = (
                match.group(0)
                .replace("\\", "/")
                .lower()
            )

            break

    if mentioned_file:
        for result in results:
            result_file = (
                result["file"]
                .replace("\\", "/")
                .lower()
            )

            if result_file.endswith(mentioned_file):
                selected_results.append(result)

    # --------------------------------------------------------
    # FALLBACK TO TOP RESULTS
    # --------------------------------------------------------

    if not selected_results:

        selected_results = results[:5]

    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for result in selected_results:

        context_parts.append(
            f"FILE: {result['file']}\n"
            f"SOURCE CODE:\n"
            f"{result['text']}"
        )

    context = (
        "\n\n"
        + ("\n\n" + "-" * 60 + "\n\n").join(
            context_parts
        )
    )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        context,
    )

    print("\nAssistant:")
    print(answer)