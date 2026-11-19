"""Skeleton training entrypoint for the triage MRE.

Kept for path-faithfulness with the traceback in logs/run_042_crash.log
(`/workspace/train.py`, forward at line ~142). Not expected to execute here:
no checkpoints, no GPU allocator to reproduce the abort.

Launch (as in the lab):
    accelerate launch --config_file accelerate_config.yaml --num_processes 4 \
        train.py --config configs/exp_failed.yaml
"""

import argparse
import os

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a configs/*.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(cfg: dict) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    lora = LoraConfig(
        r=cfg["training"]["lora_r"],
        lora_alpha=cfg["training"]["lora_alpha"],
        lora_dropout=cfg["training"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def build_dataset(tokenizer, cfg: dict):
    raw = load_dataset(cfg["dataset"], split="train")
    max_len = cfg["training"]["max_seq_length"]

    def tokenize(batch):
        text = [p + "\n" + r for p, r in zip(batch["instruction"], batch["response"], strict=False)]
        out = tokenizer(text, truncation=True, max_length=max_len, padding=False)
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    return raw.map(tokenize, batched=True, remove_columns=raw.column_names)


def forward(self, inputs):
    """Wrapped by the Trainer via `model_forward`; matches the traceback frame."""
    return self.model(**inputs)


def main() -> None:
    cfg = load_config(parse_args().config)
    training = cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_model(cfg)
    dataset = build_dataset(tokenizer, cfg)

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        gradient_checkpointing=training["gradient_checkpointing"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["lr_scheduler_type"],
        warmup_ratio=training["warmup_ratio"],
        num_train_epochs=training["num_train_epochs"],
        bf16=training["bf16"],
        logging_steps=training["logging_steps"],
        save_steps=training["save_steps"],
        optim="adamw_torch",
        ddp_find_unused_parameters=False,
        deepspeed=os.path.join("configs", cfg["deepspeed"]),
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
