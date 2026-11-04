import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_NAME = "Qwen/Qwen3-1.7B"

ADAPTER_PATH = (
    BASE_DIR
    / "crypto"
    / "llm"
    / "training"
    / "qwen3-1.7b-qlora"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("TESTING TRAINED QLORA MODEL")
print("=" * 60)

print(f"Base model : {MODEL_NAME}")
print(f"Adapter    : {ADAPTER_PATH}")
print(f"Device     : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")


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
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# BASE MODEL
# ============================================================

print("Loading base Qwen3 model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

print("Base model loaded.")


# ============================================================
# LOAD QLORA ADAPTER
# ============================================================

print("Loading trained QLoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_PATH),
)

model.eval()

print("QLoRA adapter loaded successfully.")


# ============================================================
# GENERATION FUNCTION
# ============================================================

def ask_model(question: str):

    messages = [
        {
            "role": "system",
            "content": (
                "You are the AI assistant for the Crypto Volatility "
                "Forecast and Smart Signal Generator project. "
                "Answer using only information learned from the project. "
                "Do not invent files, functions, models, metrics, "
                "or behavior."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    input_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
        )

    generated_tokens = output[0][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# TEST QUESTIONS
# ============================================================

questions = [
    "What does trading/signal_generator.py do?",
    "Which trading actions are present in trading/signal_generator.py?",
    "What classes are defined in trading/signal_generator.py?",
]


# ============================================================
# RUN TESTS
# ============================================================

print("\n")
print("=" * 60)
print("MODEL RESPONSES")
print("=" * 60)

for i, question in enumerate(questions, start=1):

    print(f"\nTEST {i}")
    print("-" * 60)
    print("QUESTION:")
    print(question)

    answer = ask_model(question)

    print("\nANSWER:")
    print(answer)


print("\n")
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)