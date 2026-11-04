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
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

PROMPT_FILE = (
    BASE_DIR
    / "crypto"
    / "llm"
    / "prompts"
    / "system_prompt.txt"
)


# ============================================================
# RAG SETUP
# ============================================================

RAG_DIR = Path(__file__).resolve().parents[1] / "rag"

sys.path.insert(0, str(RAG_DIR))

from retrieve import retrieve


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen3-1.7B"

ADAPTER_PATH = (
    BASE_DIR
    / "crypto"
    / "llm"
    / "training"
    / "qwen3-1.7B-qlora"
)


# ============================================================
# LOAD SYSTEM PROMPT
# ============================================================

print("Checking system prompt...")

if not PROMPT_FILE.exists():
    raise FileNotFoundError(
        f"System prompt not found: {PROMPT_FILE}"
    )

SYSTEM_PROMPT = PROMPT_FILE.read_text(
    encoding="utf-8"
)

print("System prompt loaded.")


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD QWEN
# ============================================================

print("Loading Qwen3-1.7B...")

if DEVICE == "cuda":

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

print("Base model loaded.")

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

def generate_answer(question, context=""):

    user_prompt = f"""
Retrieved project context:

{context}

User question:

{question}
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False
        )

    input_length = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    # Safety cleanup in case thinking tags appear
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL
    )

    answer = answer.strip()

    return answer


# ============================================================
# TEST RAG + QWEN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("QWEN + RAG PROJECT ASSISTANT TEST")
    print("=" * 60)


    # --------------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------------

    question = (
        "What does trading/signal_generator.py do? "
        "Mention only details explicitly supported by the "
        "retrieved project context."
    )


    print()
    print("Question:")
    print(question)


    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    print()
    print("Searching project knowledge...")

    results = retrieve(
        question,
        top_k=5
    )

    print()
    print(f"Retrieved {len(results)} project chunks.")


    # --------------------------------------------------------
    # DISPLAY RETRIEVED CONTEXT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)

    for i, result in enumerate(results, 1):

        print()
        print(f"--- Retrieved Chunk {i} ---")
        print(f"File: {result['file']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"])


    # --------------------------------------------------------
    # EXACT-FILE FILTERING
    # --------------------------------------------------------

    question_normalized = question.lower().replace("\\", "/")
question_normalized = question.lower().replace("\\", "/")

requested_file_results = []

for result in results:

    result_file = result["file"].replace("\\", "/").lower()

    relative_file = (
        result_file.split("crypto/", 1)[-1]
        if "crypto/" in result_file
        else result_file
    )

    if (
        result_file in question_normalized
        or relative_file in question_normalized
    ):
        requested_file_results.append(result)


    if requested_file_results:

        context_results = requested_file_results

        print()
        print(
            "Using exact-file context for the requested file."
        )

    else:

        context_results = results

        print()
        print(
            "No exact file match found. "
            "Using semantic retrieval context."
        )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(
        f"File: {result['file']}\n"
        f"{result['text']}"
        for result in context_results
    )


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    print()
    print("Generating answer...")

    answer = generate_answer(
        question,
        context
    )


    # --------------------------------------------------------
    # DISPLAY ANSWER
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("QWEN ANSWER")
    print("=" * 60)

    print(answer)


    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("RAG + QWEN TEST COMPLETED")
    print("=" * 60)