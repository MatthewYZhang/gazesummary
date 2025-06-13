# stage3_train.py
import os
import torch
import json
from transformers import Trainer, TrainingArguments, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from functools import partial

from GazeT5ForCausalLM import GazeT5ForCausalLM
# Import the main config, which now internally handles EyeT5Config details
from config import GazeT5ForCausalLMConfig

from stage23_dataset import ArticleGazeDS, collate_fn

# --- Configuration ---
LLM_NAME   = "Qwen/Qwen2.5-3B-Instruct" # Or 3B, match your base and EyeT5Config.proj_dim
GAZE_CKPT  = "./trained_models/gaze_t5_model.pt"  # Stage-1
ALIGN_CPT  = "./trained_models/stage2_align.pt"   # Stage-2 Weights for proj/pool
DATA_DIR   = "data/gaze_stage23"

SAVE_DIR_BASE = "./trained_models/stage3_gaze_qwen_final" # Main directory for this run
ADAPTER_SAVE_DIR = os.path.join(SAVE_DIR_BASE, "llm_adapters")
PROJ_POOL_SAVE_PATH = os.path.join(SAVE_DIR_BASE, "gaze_proj_pool.pt")
TOKENIZER_SAVE_DIR = SAVE_DIR_BASE # Save tokenizer in base directory
CONFIG_SAVE_PATH = os.path.join(SAVE_DIR_BASE, "inference_config.json")

# --- Instantiate the main config ---
# If you need to override specific EyeT5 parameters (e.g., proj_dim to match LLM), do it here.
# Example: Ensure proj_dim matches Qwen2.5-3B (2048) or 7B (3584)
# qwen_hidden_size = 2048 # Example for Qwen2.5-7B-Instruct, verify correct value


model_config = GazeT5ForCausalLMConfig(
    llm_name=LLM_NAME,
    gaze_ckpt_path=GAZE_CKPT,
    # eye_t5_config_overrides=eye_t5_overrides # Pass overrides dict
)

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_config.llm_name, use_fast=True, padding_side='left', trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # Update config's pad_token_id if tokenizer lacked it initially
    model_config.pad_token_id = tokenizer.eos_token_id
    print("Set pad_token to eos_token and updated config.")


# --- Instantiate Base Model ---
# Uses the model_config, which includes the overridden EyeT5 parameters
print("Instantiating GazeT5ForCausalLM model...")
model = GazeT5ForCausalLM(model_config)
print("Model instantiated.")

# --- Load Stage-2 Weights (Pool Query & Projection) ---
# (Loading logic remains the same - ensure keys "proj" and "pool_query" match your checkpoint)
try:
    print(f"Loading Stage-2 alignment weights from: {ALIGN_CPT}")
    align_state = torch.load(ALIGN_CPT, map_location="cpu")
    # Load proj weights state dict directly
    if hasattr(model.gaze_enc, 'proj') and "proj" in align_state:
        model.gaze_enc.proj.load_state_dict(align_state["proj"])
        print("Loaded Stage-2 projection weights.")
    else:
        print("Warning: 'proj' weights not found in alignment checkpoint or model structure.")

    # Load pool query
    if hasattr(model.gaze_enc, 'encoder') and hasattr(model.gaze_enc.encoder, 'pool') and "pool_query" in align_state:
        # Ensure dimensions match if loading directly into data
        if model.gaze_enc.encoder.pool.query.data.shape == align_state["pool_query"].shape:
             model.gaze_enc.encoder.pool.query.data.copy_(align_state["pool_query"])
             print("Loaded Stage-2 pool query weights.")
        else:
            print(f"Warning: Shape mismatch for 'pool_query'. Model: {model.gaze_enc.encoder.pool.query.data.shape}, Ckpt: {align_state['pool_query'].shape}")
    else:
         print("Warning: 'pool_query' not found in alignment checkpoint or model structure.")

except FileNotFoundError:
    print(f"Warning: Stage-2 alignment checkpoint not found at {ALIGN_CPT}.")
except Exception as e:
    print(f"Warning: Error loading Stage-2 alignment weights from {ALIGN_CPT}: {e}")


# --- Apply PEFT (LoRA) to the LLM part ---
# (LoRA config and application remain the same)
# Ensure target_modules are correct for your LLM_NAME
lora_cfg = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], # Verify for Qwen2.5-7B
)
model.llm = get_peft_model(model.llm, lora_cfg)
print("Applied LoRA adapters to the LLM.")
model.llm.print_trainable_parameters()

# --- Final Parameter Check ---
# (Printing trainable parameters remains the same)
print("Overall Trainable Parameters:")
# ... (print loop) ...
trainable_parameters_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_parameters = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_parameters / 1e6:.2f} M")
print(f"Trainable parameters: {trainable_parameters_total / 1e6:.2f} M")




# --- Dataset / Trainer ---
# Pass the main model_config to the dataset constructor
ds_train = ArticleGazeDS(DATA_DIR, "data/gaze_stage1", "train",
                         model_config, tokenizer)

# Collator needs the main config to access nested EyeT5 params
collate_partial = partial(collate_fn, tok=tokenizer, cfg=model_config)

train_args = TrainingArguments(
    output_dir=os.path.join(SAVE_DIR_BASE, "trainer_logs"),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    # max_steps=100, # Increase for actual training
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=50,
    fp16=True, # Or bf16=True
    # gradient_checkpointing=True,
    remove_unused_columns=False,
    save_strategy="steps",
    save_total_limit=2,
    report_to="none", # Or "wandb", "tensorboard"
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=ds_train,
    data_collator=collate_partial,
    tokenizer=tokenizer, # Pass tokenizer for saving
)

# --- Train & Save ---
print("Starting training...")
trainer.train()
print("Training finished.")

print(f"--- Saving Stage 3 Components to {SAVE_DIR_BASE} ---")
os.makedirs(SAVE_DIR_BASE, exist_ok=True)
os.makedirs(ADAPTER_SAVE_DIR, exist_ok=True)

# 8a. Save LoRA Adapters from the LLM part
try:
    model.llm.save_pretrained(ADAPTER_SAVE_DIR)
    print(f"✓ LLM Adapters saved to: {ADAPTER_SAVE_DIR}")
except Exception as e:
    print(f"ERROR saving LLM adapters: {e}")

# 8b. Save fine-tuned Projection & Pool Query from Gaze Encoder
try:
    # Ensure components exist
    if not hasattr(model, 'gaze_enc') or \
        not hasattr(model.gaze_enc, 'proj') or \
        not hasattr(model.gaze_enc, 'encoder') or \
        not hasattr(model.gaze_enc.encoder, 'pool') or \
        not hasattr(model.gaze_enc.encoder.pool, 'query'):
        raise AttributeError("Required gaze encoder components (proj/pool.query) not found.")

    proj_pool_state = {
        "proj": model.gaze_enc.proj.state_dict(),
        "pool_query": model.gaze_enc.encoder.pool.query.data.clone().cpu()
    }
    torch.save(proj_pool_state, PROJ_POOL_SAVE_PATH)
    print(f"✓ Gaze Proj/Pool state saved to: {PROJ_POOL_SAVE_PATH}")
except Exception as e:
    print(f"ERROR saving Gaze Proj/Pool state: {e}")

# 8c. Save Tokenizer
try:
    tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)
    print(f"✓ Tokenizer saved to: {TOKENIZER_SAVE_DIR}")
except Exception as e:
    print(f"ERROR saving Tokenizer: {e}")

# 8d. Save Inference Configuration File
try:
    inference_config = {
        "llm_name": model_config.llm_name,
        "gaze_stage1_ckpt_path": model_config.gaze_ckpt_path, # Path to Pre-trained Gaze Encoder
        # --- Store paths relative to the base save dir or absolute paths ---
        # Storing relative paths is often better if the whole folder is moved
        "adapter_dir": os.path.relpath(ADAPTER_SAVE_DIR, SAVE_DIR_BASE),
        "proj_pool_path": os.path.relpath(PROJ_POOL_SAVE_PATH, SAVE_DIR_BASE),
        "tokenizer_dir": os.path.relpath(TOKENIZER_SAVE_DIR, SAVE_DIR_BASE), # Or just use base dir
        # --- Store the EyeT5 config parameters needed to init EyeAutoEncoderT5 ---
        "eye_t5_config_dict": model_config.eye_t5_config_dict,
        # Add any other info needed for inference (e.g., LLM dtype used)
        "llm_dtype": "bfloat16" # Example
    }
    with open(CONFIG_SAVE_PATH, 'w') as f:
        json.dump(inference_config, f, indent=4)
    print(f"✓ Inference config saved to: {CONFIG_SAVE_PATH}")
except Exception as e:
    print(f"ERROR saving Inference config: {e}")

print("--- Stage 3 Saving Complete ---")