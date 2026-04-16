"""
Fine-tuning entry point for SEC Filing Analyst.

This script is intentionally thin — all logic lives in config.
The pattern: train.py reads config, sets up components, calls trainer.
This is how Hugging Face, Meta, and most production teams structure it.

Unsloth note: we import unsloth FIRST, before transformers.
Unsloth patches PyTorch at import time — if transformers loads first,
the patches don't apply and you lose the speed/memory benefits.
This ordering requirement is documented in unsloth's repo and is a
common source of "why is my GPU usage still high?" bugs.
"""

# ── CRITICAL: unsloth must be imported before transformers ──────────────────
from unsloth import FastLanguageModel
import torch
import yaml
import wandb
from pathlib import Path
from datasets import load_from_disk
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_model(cfg: dict):
    """
    Load base model with QLoRA quantization via Unsloth.
    
    Why Unsloth's FastLanguageModel over vanilla HuggingFace?
    
    Unsloth rewrites the attention kernels using Triton (same as 
    FlashAttention but works on more GPU types including T4).
    Result: 2x faster training, 40% less VRAM for the same model.
    
    On Kaggle T4:
    - Without Unsloth: Mistral 7B QLoRA barely fits, ~45 min/epoch
    - With Unsloth:    Mistral 7B QLoRA fits comfortably, ~22 min/epoch
    
    The API is a drop-in replacement for AutoModelForCausalLM +
    get_peft_model(). Same interface, better performance.
    """
    model_cfg = cfg["model"]
    quant_cfg = cfg["quantization"]
    lora_cfg  = cfg["lora"]
    
    # Compute dtype
    dtype_map = {"bfloat16": torch.bfloat16, 
                 "float16": torch.float16, 
                 "float32": torch.float32}
    compute_dtype = dtype_map[model_cfg["dtype"]]
    
    print(f"Loading {model_cfg['base_model']} with 4-bit quantization...")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=compute_dtype,
        load_in_4bit=quant_cfg["load_in_4bit"],
        # token= — add HF token here if model is gated
    )
    
    print(f"Base model loaded. Applying LoRA adapters...")
    
    # Apply LoRA via Unsloth (equivalent to get_peft_model but optimized)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        use_gradient_checkpointing="unsloth",
        # "unsloth" = Unsloth's optimized gradient checkpointing
        # vs True = standard HuggingFace gradient checkpointing
        # Unsloth's version is ~10% faster and uses less VRAM
        random_state=42,
    )
    
    # Print parameter counts — this is the "trainable params" number
    # you put in your README and mention in interviews
    trainable = sum(p.numel() for p in model.parameters() 
                    if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable:,} "
          f"({100*trainable/total:.3f}% of total)")
    print(f"Total parameters:     {total:,}")
    # Expected output:
    # Trainable parameters: ~40,000,000 (0.5% of total)
    # Total parameters:     ~7,200,000,000
    
    return model, tokenizer


def setup_trainer(model, tokenizer, train_dataset, val_dataset, 
                  cfg: dict) -> SFTTrainer:
    """
    Configure and return the SFTTrainer.
    
    SFTTrainer (Supervised Fine-Tuning Trainer) from TRL wraps
    HuggingFace Trainer with LLM-specific defaults:
    - Handles the text→tokens→loss pipeline for causal LM
    - Supports dataset_text_field for direct text column training
    - Built-in packing, completion-only masking, etc.
    
    TrainingArguments vs SFTConfig:
    In trl>=0.8, SFTConfig is the preferred way — it merges
    TrainingArguments + SFT-specific args into one object.
    We use SFTConfig because your installed trl==0.24.0 supports it.
    """
    train_cfg = cfg["training"]
    eval_cfg  = cfg["evaluation"]
    log_cfg   = cfg["logging"]
    out_cfg   = cfg["output"]
    
    # Total steps for scheduler — needed to understand warmup_steps
    steps_per_epoch = (len(train_dataset) // 
                       (train_cfg["per_device_train_batch_size"] * 
                        train_cfg["gradient_accumulation_steps"]))
    total_steps = steps_per_epoch * train_cfg["num_train_epochs"]
    print(f"\nEstimated total training steps: {total_steps}")
    print(f"Steps per epoch: {steps_per_epoch}")
    
    sft_config = SFTConfig(
        # Output
        output_dir=out_cfg["output_dir"],
        
        # Training duration
        num_train_epochs=train_cfg["num_train_epochs"],
        
        # Batch & accumulation
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        
        # Learning rate
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        
        # Optimizer & precision
        optim=train_cfg["optim"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        max_grad_norm=train_cfg["max_grad_norm"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        
        # Sequence
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_text_field="text",
        # "text" = the column in our dataset that has the full ChatML string
        # SFTTrainer tokenizes this column and trains on it end-to-end
        packing=train_cfg["packing"],
        
        # Evaluation
        eval_strategy=eval_cfg["eval_strategy"],
        eval_steps=eval_cfg["eval_steps"],
        save_strategy=eval_cfg["save_strategy"],
        save_steps=eval_cfg["save_steps"],
        load_best_model_at_end=eval_cfg["load_best_model_at_end"],
        metric_for_best_model=eval_cfg["metric_for_best_model"],
        
        # Logging
        logging_steps=log_cfg["logging_steps"],
        report_to=log_cfg["report_to"],
        run_name=log_cfg["run_name"],
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
            # If eval_loss doesn't improve for 3 consecutive eval points,
            # stop training. Prevents overfitting on small datasets.
            # Patience=3 at eval_steps=50 means we tolerate 150 steps
            # without improvement — generous enough to avoid premature stops.
        ],
    )
    
    return trainer


def main():
    # Load config
    cfg = load_config("configs/training_config.yaml")
    
    # Initialize W&B — do this before model loading so the full
    # training run is captured including model loading time
    wandb.init(
        project="sec-filing-analyst",
        name=cfg["logging"]["run_name"],
        config=cfg,  # logs entire config to W&B — fully reproducible
        tags=["qlora", "mistral-7b", "sec", "finance"]
    )
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_from_disk("data/dataset/train")
    val_dataset   = load_from_disk("data/dataset/validation")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Setup model
    model, tokenizer = setup_model(cfg)
    
    # Setup trainer
    trainer = setup_trainer(
        model, tokenizer, 
        train_dataset, val_dataset, 
        cfg
    )
    
    # Train
    print("\nStarting training...")
    print("Watch your W&B dashboard: https://wandb.ai")
    
    trainer_stats = trainer.train()
    
    print(f"\nTraining complete.")
    print(f"Runtime: {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"Samples/sec: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    
    # Save the LoRA adapter (NOT the full model — just the delta)
    # The adapter is ~100MB. The merged model would be ~14GB.
    # We save the adapter and merge only for deployment.
    adapter_path = Path(cfg["output"]["output_dir"]) / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nLoRA adapter saved → {adapter_path}")
    
    # Log final metrics to W&B
    wandb.log({"training_complete": True})
    wandb.finish()
    
    return trainer_stats


if __name__ == "__main__":
    main()