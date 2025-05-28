# inference.py (Revised for separate component loading)
import os
import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import time

# --- Required Imports ---
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel # To load adapters

# Assuming these are in the path and correctly defined
from config import EyeT5Config
from gazeT5 import EyeAutoEncoderT5

# Assuming dataset loading helpers are available
from stage23_dataset import ArticleGazeDS # For potential data structure reference
from datasets import load_from_disk # If using HF datasets

# --- Helper to prepare ONE gaze input data point (Similar to collate_fn) ---
def prepare_gaze_input_for_inference(gaze_tensors_list, gaze_masks_list, sentences_list, eye_t5_cfg, device):
    """ Pads gaze data for a single example (batch size 1). """
    max_sents = len(sentences_list)
    max_len = 0
    if gaze_tensors_list:
        valid_lengths = [len(t) for t in gaze_tensors_list if hasattr(t, '__len__') and len(t) > 0]
        if valid_lengths:
            max_len = max(valid_lengths)

    in_dim_numeric = eye_t5_cfg.in_dim_numeric

    gaze_num_padded  = torch.zeros(1, max_sents, max_len, in_dim_numeric, device=device)
    gaze_mask_padded = torch.zeros(1, max_sents, max_len, dtype=torch.bool, device=device)

    for j, (x_num, x_mask) in enumerate(zip(gaze_tensors_list, gaze_masks_list)):
        if isinstance(x_num, torch.Tensor):
            L = len(x_num)
            if L > 0 and x_num.shape[1] == in_dim_numeric:
                 gaze_num_padded[0, j, :L]  = x_num.to(device) # Ensure data is on correct device
                 gaze_mask_padded[0, j, :L] = x_mask.to(device)

    sentences_batch = [sentences_list] # Batch size 1 list of lists

    return gaze_num_padded, gaze_mask_padded, sentences_batch

# --- Main Inference Function ---
def main(args):
    # --- 1. Load Inference Configuration ---
    print(f"--- Loading Inference Config from {args.model_base_dir} ---")
    config_path = os.path.join(args.model_base_dir, "inference_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"inference_config.json not found in {args.model_base_dir}")
    with open(config_path, 'r') as f:
        inference_config = json.load(f)
    print(f"Inference Config: {json.dumps(inference_config, indent=2)}")

    # Determine absolute paths for components based on model_base_dir
    adapter_dir = os.path.join(args.model_base_dir, inference_config["adapter_dir"])
    proj_pool_path = os.path.join(args.model_base_dir, inference_config["proj_pool_path"])
    tokenizer_dir = os.path.join(args.model_base_dir, inference_config.get("tokenizer_dir", ".")) # Default to base if key missing
    gaze_stage1_ckpt_path = inference_config["gaze_stage1_ckpt_path"] # Assume this is absolute or relative to execution
    llm_name = inference_config["llm_name"]
    llm_dtype_str = inference_config.get("llm_dtype", "float16") # Default dtype
    llm_torch_dtype = torch.bfloat16 if llm_dtype_str == "bfloat16" else torch.float16

    # --- 2. Load Tokenizer ---
    print(f"--- Loading Tokenizer from {tokenizer_dir} ---")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, padding_side='left', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token.")

    # --- 3. Load Base LLM (with potential quantization/device_map) ---
    print(f"--- Loading Base LLM: {llm_name} ---")
    # Example: Load with device_map and quantization
    # bnb_config = BitsAndBytesConfig(load_in_8bit=True) # Or 4bit
    base_llm = AutoModelForCausalLM.from_pretrained(
        llm_name,
        # quantization_config=bnb_config, # Apply quantization if needed
        device_map="auto", # Use accelerate for auto device placement
        torch_dtype=llm_torch_dtype, # Set desired dtype
        trust_remote_code=True
    )
    print(f"Base LLM loaded on device(s): {base_llm.device}") # Might show 'meta' if using device_map

    # --- 4. Load and Apply LoRA Adapters ---
    print(f"--- Loading Adapters from {adapter_dir} ---")
    # Load adapters onto the base model. PEFT handles device placement matching base_llm.
    llm = PeftModel.from_pretrained(base_llm, adapter_dir)
    print("LoRA adapters loaded.")
    # Optional: Merge adapters if you don't need to switch them later
    # print("Merging adapters...")
    # llm = llm.merge_and_unload()
    # print("Adapters merged.")
    llm.eval()


    # --- 5. Load Gaze Encoder (EyeAutoEncoderT5) ---
    print("--- Loading Gaze Encoder ---")
    eye_t5_config_dict = inference_config["eye_t5_config_dict"]
    eye_t5_cfg = EyeT5Config(**eye_t5_config_dict)
    gaze_enc = EyeAutoEncoderT5(eye_t5_cfg)
    print("Gaze Encoder instantiated.")

    # 5a. Load Stage 1 Weights
    print(f"Loading Stage 1 weights from: {gaze_stage1_ckpt_path}")
    if not os.path.isfile(gaze_stage1_ckpt_path):
        raise FileNotFoundError(f"Stage 1 Gaze checkpoint not found: {gaze_stage1_ckpt_path}")
    state1 = torch.load(gaze_stage1_ckpt_path, map_location="cpu")["model_state"]
    gaze_enc.load_state_dict(state1, strict=False)
    print("Stage 1 weights loaded.")

    # 5b. Load Stage 3 Fine-tuned Proj/Pool Weights
    print(f"Loading Stage 3 Proj/Pool weights from: {proj_pool_path}")
    if not os.path.isfile(proj_pool_path):
        raise FileNotFoundError(f"Stage 3 Proj/Pool checkpoint not found: {proj_pool_path}")
    state3_proj_pool = torch.load(proj_pool_path, map_location="cpu")
    # Load proj state dict
    if hasattr(gaze_enc, 'proj') and "proj" in state3_proj_pool:
        gaze_enc.proj.load_state_dict(state3_proj_pool["proj"])
    else: print("Warning: Gaze encoder proj state not loaded from Stage 3 checkpoint.")
    # Load pool query tensor data
    if hasattr(gaze_enc, 'encoder') and hasattr(gaze_enc.encoder, 'pool') and "pool_query" in state3_proj_pool:
        if gaze_enc.encoder.pool.query.data.shape == state3_proj_pool["pool_query"].shape:
            gaze_enc.encoder.pool.query.data.copy_(state3_proj_pool["pool_query"])
        else: print("Warning: Shape mismatch for pool_query in Stage 3 checkpoint.")
    else: print("Warning: Gaze encoder pool query state not loaded from Stage 3 checkpoint.")
    print("Stage 3 Proj/Pool weights loaded.")

    # 5c. Move Gaze Encoder to device & eval mode
    # Usually run gaze encoder on CPU or a single GPU, as it's smaller
    # Let's default to CPU unless a specific GPU is desired
    gaze_enc_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Or "cpu"
    gaze_enc.to(gaze_enc_device)
    gaze_enc.eval()
    print(f"Gaze Encoder moved to {gaze_enc_device} and set to eval mode.")


    # --- 6. Load Dataset ---
    print(f"--- Loading Data ---")
    print(f"Attempting to load HF dataset using ArticleGazeDS...")
    print(f"  Article data dir: {args.article_data_dir}")
    print(f"  Sentence data dir: {args.sentence_data_dir}")
    print(f"  Split: {args.split}")
    try:
        # Instantiate ArticleGazeDS
        # It needs the config mainly for in_dim_numeric if handling empty gaze in __getitem__
        # We pass eye_t5_cfg reconstructed from the inference config
        dataset = ArticleGazeDS(
            article_hf_dir=args.article_data_dir,
            sentence_hf_dir=args.sentence_data_dir,
            split=args.split,
            cfg=eye_t5_cfg, # Pass the reconstructed EyeT5Config
            tokenizer=tokenizer # Pass tokenizer (though likely not used in __getitem__)
        )
        raw_data = dataset # Assign the dataset object to raw_data
        print(f"✓ Successfully loaded HF dataset split '{args.split}' with {len(raw_data)} examples.")

    except Exception as e:
        print(f"ERROR: Failed to load HF dataset using ArticleGazeDS: {e}")
        print("NOTE: JSON file loading is not implemented in this version.")
        # Optionally, implement JSON loading here as a fallback if needed
        # For now, we will exit if HF dataset loading fails
        return # Exit if data loading failed

    if not raw_data:
        print("ERROR: No data loaded. Exiting.")
        return


    start_time = time.time()
    # --- 7. Run Inference Loop ---
    results_gaze = {}
    print("--- Starting Inference ---")
    llm_input_device = llm.device # Device for LLM inputs

        # Iterate directly over the dataset object
    for i, example in enumerate(tqdm(raw_data)):
        # 'example' is now a dictionary returned by ArticleGazeDS.__getitem__
        # break
        try:
            # 7a. Extract data from the example dictionary
            # These keys should match the output of ArticleGazeDS.__getitem__
            gaze_tensors_list = example.get("gaze_nums_list", [])
            gaze_masks_list = example.get("gaze_masks_list", [])
            sentences_list = example.get("sentences_list", [])
            # Construct prompt text from sentences (as done in __getitem__)
            article_text = " ".join(sentences_list)
            prompt_text = (f"Please summarize the following article using one paragraph with around 150 words with personalization based on the user's gaze data:\n\n{article_text}")
            # Get article ID - assuming the underlying dataset loaded by
            # ArticleGazeDS has this field accessible in its __getitem__
            # This might require adjusting ArticleGazeDS if it doesn't pass it through
            # Or accessing the original dataset item if ArticleGazeDS stores the index `idx`
            article_id = example.get("article_id", f"item_{i}") # Placeholder if ID not in example dict


            # 7b. Prepare Gaze Input and Calculate Gaze Embeddings
            gaze_num_padded, gaze_mask_padded, sentences_batch = prepare_gaze_input_for_inference(
                gaze_tensors_list, gaze_masks_list, sentences_list, eye_t5_cfg, gaze_enc_device
            )
            gaze_num_padded = gaze_num_padded.reshape(-1, gaze_num_padded.shape[-2], gaze_num_padded.shape[-1]) # Reshape to [B*S, K, dm]
            gaze_mask_padded = gaze_mask_padded.reshape(gaze_mask_padded.shape[-2], gaze_mask_padded.shape[-1]) # Reshape to [B*S, K]
            with torch.no_grad():
                # Gaze encoder forward pass
                _, flat_tok = gaze_enc(gaze_num_padded.to(gaze_enc_device),
                                     gaze_mask_padded.to(gaze_enc_device),
                                     sum(sentences_batch, []))
                L_gaze = flat_tok.shape[0] * flat_tok.shape[1]
                gaze_tok = flat_tok.reshape(1, L_gaze, -1)
                gaze_tok = gaze_tok.to(device=llm_input_device, dtype=llm_torch_dtype)

            # 7c. Prepare Prompt Input and Calculate Prompt Embeddings
            msgs = [{"role": "user", "content": prompt_text}]
            prompt_input_ids = tokenizer.apply_chat_template(
                msgs,
                add_generation_prompt=True, # Important for inference
                return_tensors="pt"
            ).to(llm_input_device) # Move input_ids tensor to LLM device

            # --- Create the attention mask for the prompt manually ---
            # Since there's no padding in this single sequence, mask is all ones.
            prompt_attention_mask = torch.ones_like(prompt_input_ids, device=llm_input_device)

            # Calculate prompt embeddings using the input IDs
            with torch.no_grad():
                prompt_emb = llm.get_input_embeddings()(prompt_input_ids) # Shape: [1, L_prompt, D]

            # 7d. Combine Embeddings and Create Full Attention Mask
            # Concatenate gaze embeddings and prompt embeddings
            inputs_embeds = torch.cat([gaze_tok, prompt_emb], dim=1) # Shape: [1, L_gaze + L_prompt, D]

            # Create attention mask for the gaze part
            gaze_att = torch.ones(gaze_tok.shape[:2], dtype=torch.long, device=llm_input_device) # Shape: [1, L_gaze]

            # Concatenate gaze attention mask and the manually created prompt attention mask
            full_attention_mask = torch.cat([gaze_att, prompt_attention_mask], dim=1) # Shape: [1, L_gaze + L_prompt]
            if inputs_embeds.dtype != llm.dtype:
                inputs_embeds = inputs_embeds.to(llm.dtype)

            # 7e. Generate Summary using the LLM
            with torch.no_grad():
                outputs = llm.generate(
                    inputs_embeds=inputs_embeds,        # Combined gaze+prompt embeddings
                    attention_mask=full_attention_mask, # Combined gaze+prompt mask
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    # Add other generation params (do_sample, temperature, etc.) if needed
                )

            # 7f. Decode Output
            # ... (decoding as before) ...
            summary_gaze = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results_gaze[article_id] = {"article": article_text, "summary_gaze": summary_gaze, article_id: article_id}
        except Exception as e:
            article_id = example.get("article_id", f"item_{i}") # Use placeholder if ID fails
            print(f"\nError processing article {article_id}: {e}")
            import traceback
            traceback.print_exc()
            results_gaze[article_id] = {"article": article_text, "summary_gaze": f"ERROR: {e}"}



        # break


    # baseline eval
    baseline_llm_name = "Qwen/Qwen2.5-3B-Instruct" # As requested
    print(f"--- Loading Baseline LLM: {baseline_llm_name} ---")
    try:
        # Load with similar settings as the gaze base model for fair comparison
        llm_base = AutoModelForCausalLM.from_pretrained(
            baseline_llm_name,
            device_map="auto", # Use auto device map
            torch_dtype=llm_torch_dtype, # Use same dtype if possible
            trust_remote_code=True
        )
        llm_base.eval()
        print(f"Baseline LLM loaded on device(s): {llm_base.device}")
        baseline_llm_device = llm_base.device
    except Exception as e:
        print(f"ERROR: Failed to load baseline model '{baseline_llm_name}': {e}")
        print("Baseline comparison will be skipped.")
        llm_base = None # Set to None if loading fails


    results_base = {}

    if llm_base is not None: # Only run if baseline model loaded successfully
        print("--- Starting Inference (Baseline Model) ---")
        for i, example in enumerate(tqdm(raw_data, desc="Baseline Model Inference")):
            # print(example)
            article_id = example.get("article_id", f"item_{i}")
            # Ensure we have the article text (also available from results_gaze if already processed)
            if article_id in results_gaze:
                 article_text = results_gaze[article_id]["article"]
            else: # Fallback if gaze model failed early for this item
                 sentences_list = example.get("sentences_list", [])
                 article_text = " ".join(sentences_list)

            try:
                # 7a. Prepare Standard Prompt (no gaze info)
                # Use a simple prompt, adjust if needed
                prompt_text_base = f"Please summarize the following article using one paragraph with around 150 words:\n{article_text}\n\nSummary:\n"
                # --- Tokenize the input FOR THE BASELINE MODEL ---
                # Keep the output as a dictionary to easily get input_ids length
                inputs_base = tokenizer(
                    prompt_text_base,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2500 # Ensure prompt fits context
                ).to(baseline_llm_device)
                # Or use apply_chat_template if baseline expects it
                # inputs_base = tokenizer.apply_chat_template(msgs_base, add_generation_prompt=True, return_tensors="pt").to(baseline_llm_device)

                # 7b. Generate Summary (Baseline Model) - Use standard input_ids
                with torch.no_grad():
                    outputs_base = llm_base.generate(
                        **inputs_base, # Pass dict containing input_ids and attention_mask
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                # 7c. Decode Output
                # Slice off prompt tokens if tokenizer includes them in output (depends on generation kwargs)
                # It's often safer to decode all and potentially clean later if needed
                # Or slice based on input length: input_len = inputs_base['input_ids'].shape[1]
                # summary_base = tokenizer.decode(outputs_base[0, input_len:], skip_special_tokens=True)
                input_token_length = inputs_base['input_ids'].shape[1]

                generated_token_ids = outputs_base[0, input_token_length:]

                # --- Decode only the generated token IDs ---
                summary_base = tokenizer.decode(generated_token_ids, skip_special_tokens=True)



                # Store result
                results_base[article_id] = {"summary_base": summary_base}

            except Exception as e:
                # print(f"\nError processing baseline model for article {article_id}: {e}")
                # traceback.print_exc()
                results_base[article_id] = {"summary_base": f"ERROR: {e}"}
            
            # break
    else:
        print("Skipping baseline model inference as it failed to load.")


    print("--- Combining Results ---")
    final_results = []
    all_ids = set(results_gaze.keys()) | set(results_base.keys()) # Get all unique IDs

    for article_id in sorted(list(all_ids)): # Process in sorted order
        gaze_data = results_gaze.get(article_id, {})
        base_data = results_base.get(article_id, {})

        # Get article text preferentially from gaze results, or reconstruct if needed
        article_text = gaze_data.get("article", "Error: Article text missing")
        if article_text == "Error: Article text missing":
             # Try to find original example if gaze processing failed early (less efficient)
             try:
                 original_example = next(ex for i, ex in enumerate(raw_data) if ex.get("article_id", f"item_{i}") == article_id)
                 article_text = " ".join(original_example.get("sentences_list", []))
             except StopIteration:
                 article_text = "Error: Article text could not be retrieved"


        final_results.append({
            "article_id": article_id,
            "article": article_text,
            "summary_gaze": gaze_data.get("summary_gaze", "N/A"),
            "summary_base": base_data.get("summary_base", "N/A") # Provide default if baseline failed/skipped
        })

    # --- 9. Save Combined Results ---
    output_path = os.path.join(args.output_dir, "generated_summaries_comparison.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(final_results).to_csv(output_path, index=False)
    end_time = time.time()
    print(f"[✓] Saved combined results for {len(final_results)} examples to {output_path}")
    print(f"Total Inference Time: {end_time - start_time:.2f} seconds")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--article_data_dir", type=str, required=True, help="Directory containing the Hugging Face dataset for articles (Stage 2/3 format).")
    parser.add_argument("--sentence_data_dir", type=str, required=True, help="Directory containing the Hugging Face dataset for sentences and gaze (Stage 1 format).")
    parser.add_argument("--split", type=str, default="test", help="Data split to load (e.g., 'test', 'validation').")
    # --- Kept Arguments ---
    # Changed argument to point to the base directory where components were saved
    parser.add_argument("--model_base_dir", type=str, required=True, help="Base directory containing saved Stage 3 components (adapters, proj/pool, config)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save output CSV")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens to generate for summary")
    args = parser.parse_args()
    main(args)

# Example Command:
'''
python inference.py \
  --article_data_dir ./data/gaze_stage23 \
  --sentence_data_dir ./data/gaze_stage1 \
  --split test \
  --model_base_dir ./trained_models/stage3_gaze_qwen_final \
  --output_dir ./results_stage3_separated \
  --max_new_tokens 450

python inference.py \
  --article_data_dir ./user_real_data/stage23 \
  --sentence_data_dir ./user_real_data/stage1 \
  --split test \
  --model_base_dir ./trained_models/stage3_gaze_qwen_final \
  --output_dir ./results_real_stage3_separated \
  --max_new_tokens 450

'''