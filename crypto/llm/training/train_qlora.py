import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_PATH = BASE_DIR / "crypto" / "llm" / "dataset" / "train.jsonl"
OUTPUT_DIR = BASE_DIR / "crypto" / "llm" / "training" / "qwen3-1.7b-qlora"

MODEL_NAME = "Qwen/Qwen3-1.7B"


# ============================================================
# BASIC CHECKS
# ============================================================

print("=" * 60)
print("QLoRA TRAINING SETUP")
print("=" * 60)

print(f"Dataset : {DATASET_PATH}")
print(f"Output  : {OUTPUT_DIR}")
print(f"Model   : {MODEL_NAME}")

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

if torch.cuda.is_available():
    print(f"CUDA    : True")
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
else:
    print("CUDA    : False")

print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH),
    split="train",
)

print(f"Dataset examples: {len(dataset)}")


# ============================================================
# CONVERT CHAT MESSAGES TO TEXT
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


dataset = dataset.map(format_example)

print("Chat formatting: OK")


# ============================================================
# TOKENIZATION
# ============================================================

MAX_LENGTH = 384


def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)

print("Tokenization: OK")


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
# LOAD MODEL
# ============================================================

print("Loading Qwen3 in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

model.config.use_cache = False

print("Model loaded successfully.")


# ============================================================
# PREPARE MODEL FOR QLORA
# ============================================================

model = prepare_model_for_kbit_training(model)


# ============================================================
# LoRA CONFIGURATION
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=2,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    fp16=True,

    logging_steps=5,
    save_strategy="epoch",

    gradient_checkpointing=True,

    optim="paged_adamw_8bit",

    report_to="none",

    remove_unused_columns=False,
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


# ============================================================
# TRAIN
# ============================================================

print("=" * 60)
print("STARTING QLoRA TRAINING")
print("=" * 60)

train_result = trainer.train()

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print(f"Model saved to: {OUTPUT_DIR}")
print("=" * 60)