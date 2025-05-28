# stage2_train.py
import torch
import os
import argparse
import json
from functools import partial
from transformers import Trainer, TrainingArguments, AutoTokenizer

# Import the refactored model and its config
from GazeT5ForCausalLM import GazeT5ForCausalLM
from config import GazeT5ForCausalLMConfig # Use the main config class

# Import the updated dataset and collator
from stage23_dataset import ArticleGazeDS, collate_fn

# --- Constants ---
# Ensure LLM_NAME matches the one intended for the final model
# and that its hidden size corresponds to EyeT5Config's proj_dim
LLM_NAME   = "Qwen/Qwen2.5-3B-Instruct" # Or 3B? Match Stage 3 base
GAZE_CKPT  = "./trained_models/gaze_t5_model.pt"  # Path to Stage-1 Gaze Encoder weights
DATA_DIR   = "data/gaze_stage23"                  # HF Dataset directory for articles
SENT_DIR   = "data/gaze_stage1"                   # HF Dataset directory for sentences+gaze
OUTPUT_DIR = "./trained_models"                   # Directory to save Stage 2 output
OUTPUT_FILENAME = "stage2_align.pt"               # Name for the saved weights file

def main():
    # --- 1. Configuration ---
    print("--- Configuring Stage 2 ---")
    # Instantiate the main configuration class
    # Allow potential overrides, e.g., if proj_dim needs setting
    model_config = GazeT5ForCausalLMConfig(
        llm_name=LLM_NAME,
        gaze_ckpt_path=GAZE_CKPT,
        # eye_t5_config_overrides=eye_t5_overrides
    )
    # Extract the specific EyeT5 config part for easy access to S2 hyperparameters
    eye_t5_cfg = model_config.get_eye_t5_config()
    print(f"Using LLM: {model_config.llm_name}")
    print(f"Loading Stage 1 Gaze Encoder from: {model_config.gaze_ckpt_path}")
    print(f"EyeT5 Config used: {model_config.eye_t5_config_dict}")
    print(f"Stage 2 LR: {eye_t5_cfg.lr_s2}, Weight Decay: {eye_t5_cfg.weight_decay_s2}")

    # --- 2. Tokenizer ---
    print("--- Loading Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.llm_name,
        use_fast=True,
        padding_side='left', # Important for decoder models
        trust_remote_code=True # Often needed for models like Qwen
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model_config.pad_token_id = tokenizer.eos_token_id # Update config if needed
        print("Set tokenizer pad_token to eos_token.")

    # --- 3. Model Initialization ---
    print("--- Initializing Model ---")
    # Instantiating the model automatically loads LLM and Stage 1 Gaze Encoder,
    # and sets requires_grad flags correctly for Stage 2 (only proj/pool trainable).
    model = GazeT5ForCausalLM(model_config)
    # model = model.cuda() # Move model to GPU if not using device_map/Trainer handles it

    # Verify trainable parameters (should only be gaze_enc.proj and pool.query)
    print("--- Verifying Trainable Parameters for Stage 2 ---")
    trainable_params = []
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            print(f"- Trainable: {name} ({param.numel()})")
            trainable_params.append(param)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f} M")

    # Check if the expected layers are indeed trainable
    if not model.gaze_enc.proj.weight.requires_grad:
        print("Warning: Gaze encoder projection weight is NOT trainable!")
    if not model.gaze_enc.encoder.pool.query.requires_grad:
         print("Warning: Gaze encoder pool query is NOT trainable!")


    # --- 4. Dataset and Collator ---
    print("--- Loading Dataset ---")
    # Use the main model_config here
    train_dataset = ArticleGazeDS(article_hf_dir=DATA_DIR,
                                  sentence_hf_dir=SENT_DIR,
                                  split="train", # Assuming 'train' split exists
                                  cfg=model_config,
                                  tokenizer=tokenizer)

    # Use partial to bind the tokenizer and config to the collate function
    data_collator = partial(collate_fn, tok=tokenizer, cfg=model_config)
    print(f"Loaded training dataset with {len(train_dataset)} examples.")


    # --- 5. Training Arguments ---
    print("--- Setting Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "stage2_trainer_logs"), # Dir for logs/checkpoints
        per_device_train_batch_size=1,       # Adjust based on GPU memory
        gradient_accumulation_steps=4,       # Adjust based on GPU memory & desired batch size
        # warmup_steps=200,                  # Number of warmup steps
        num_train_epochs=10,                 # Or use max_steps
        # max_steps=20,                    # Alternative to epochs
        learning_rate=eye_t5_cfg.lr_s2,      # Use LR defined for Stage 2
        weight_decay=eye_t5_cfg.weight_decay_s2, # Use WD defined for Stage 2
        logging_steps=20,                    # How often to log training loss
        save_strategy="no",                  # Do not save checkpoints using HF Trainer
        fp16=True,                           # Enable mixed-precision training
        # bf16=True,                         # Use bfloat16 if supported and desired
        remove_unused_columns=False,         # Keep custom columns for the collator
        report_to="none",                    # Disable external logging (e.g., wandb) unless needed
        # gradient_checkpointing=True,       # Can save memory but slows down training
    )

    # --- 6. Trainer Initialization ---
    print("--- Initializing Trainer ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer, # Pass tokenizer even if not saving model via Trainer
        # Optimizers are typically handled by Trainer, using AdamW by default
    )

    # --- 7. Training ---
    print("--- Starting Stage 2 Training ---")
    trainer.train()
    print("--- Stage 2 Training Finished ---")

    # --- 8. Saving Trained Components ---
    print("--- Saving Stage 2 Trained Weights ---")
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ensure the required components exist before trying to save them
    if not hasattr(model, 'gaze_enc') or not hasattr(model.gaze_enc, 'proj'):
        print("Error: Cannot save projection weights, model.gaze_enc.proj not found.")
        return
    if not hasattr(model.gaze_enc, 'encoder') or not hasattr(model.gaze_enc.encoder, 'pool') or not hasattr(model.gaze_enc.encoder.pool, 'query'):
        print("Error: Cannot save pool query, model.gaze_enc.encoder.pool.query not found.")
        return

    # Save only the state_dict of the projection layer and the pool query tensor
    # Clone and move to CPU for safety, especially if trained on GPU
    proj_state_dict = model.gaze_enc.proj.state_dict()
    pool_query_tensor = model.gaze_enc.encoder.pool.query.data.clone().cpu()

    # Save relevant config part for reference during Stage 3 loading
    saved_data = {
        "eye_t5_config_dict": model_config.eye_t5_config_dict, # Save params used
        "proj": proj_state_dict,
        "pool_query": pool_query_tensor
    }

    try:
        torch.save(saved_data, save_path)
        print(f"✓ Stage 2 checkpoint (projection & pool query) saved to: {save_path}")
    except Exception as e:
        print(f"Error saving Stage 2 checkpoint: {e}")

if __name__ == "__main__":
    # Add argument parsing here if needed (e.g., override config paths, LR, etc.)
    # parser = argparse.ArgumentParser()
    # parser.add_argument(...)
    # args = parser.parse_args()
    main()