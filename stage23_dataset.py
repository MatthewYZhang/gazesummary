# stage23_dataset.py

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any
# Import the main config which contains the EyeT5Config details
from config import GazeT5ForCausalLMConfig, EyeT5Config

# ---------------------------------------------------------------------------
# ArticleGazeDS Class (Assuming this remains as previously defined)
# ---------------------------------------------------------------------------
class ArticleGazeDS(Dataset):
    """
    Loads article, summary, and associated sentence-level gaze data.
    One item corresponds to one article.
    """
    def __init__(self,
                 article_hf_dir : str,
                 sentence_hf_dir: str,
                 split: str,
                 # cfg is mainly for reference here, tokenizer is used
                 cfg : GazeT5ForCausalLMConfig,
                 tokenizer : PreTrainedTokenizer):
        # Consider loading only necessary columns to save memory if datasets are large
        self.art_ds = load_from_disk(article_hf_dir)[split]
        # Load all sentence data; could be filtered later if needed
        self.sent_ds = load_from_disk(sentence_hf_dir)[split]
        self.cfg = cfg # Store config if needed for item processing
        self.tok = tokenizer # Store tokenizer if needed for item processing

    def __len__(self):
        return len(self.art_ds)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        Processes one article to extract prompt, labels, and gaze data.
        Returns raw data structures; collation handles tokenization and padding.
        """
        art = self.art_ds[idx] # Get article metadata (summary, sentence indices)
        gaze_tensors, gaze_masks, sent_texts = [], [], []

        # Iterate through the sentence indices belonging to this article
        for sent_idx in art["sentence_indices"]:
            s = self.sent_ds[sent_idx] # Get sentence data (text, gaze coordinates)

            # Extract and structure gaze data - ensure dtype is float for numbers
            # Use .get() for robustness if keys might be missing
            gaze_coords = list(zip(s.get("gaze_x", []),
                                   s.get("gaze_y", []),
                                   s.get("gaze_dur", [])))

            # Handle cases with no gaze data for a sentence
            if gaze_coords:
                 x_num = torch.tensor(gaze_coords, dtype=torch.float)
                 gaze_tensors.append(x_num)
                 gaze_masks.append(torch.ones(len(x_num), dtype=torch.bool))
            else:
                 # Append empty tensors with correct feature dimension if no gaze
                 # Get in_dim_numeric from config safely
                 eye_cfg = self.cfg.get_eye_t5_config()
                 in_dim_numeric = eye_cfg.in_dim_numeric
                 gaze_tensors.append(torch.empty((0, in_dim_numeric), dtype=torch.float))
                 gaze_masks.append(torch.empty((0,), dtype=torch.bool))

            sent_texts.append(s.get("sentence", "")) # Append sentence text

        # Combine sentences to form the article text part of the prompt
        article_text = " ".join(sent_texts)

        # Define the prompt structure (adjust as needed)
        prompt = (f"Please summarize the following article with personalization "
                  f"based on the user's gaze data:\n\n{article_text}")

        # Return components needed for the collate function
        return {
            "prompt_text": prompt,          # The text prompt for the user turn
            "summary_text": art["summary"], # The target summary string (will be tokenized later)
            "gaze_nums_list":  gaze_tensors, # List of [Lg, F] tensors per sentence
            "gaze_masks_list": gaze_masks,  # List of [Lg] bool tensors per sentence
            "sentences_list":  sent_texts,  # List of sentence strings for this article
            "article_id": art["article_id"], # Article ID for reference
        }


# ---------------------------------------------------------------------------
# Collator Function (Revised and Complete)
# ---------------------------------------------------------------------------
def collate_fn(batch: List[Dict[str, Any]],
               tok: PreTrainedTokenizer,
               cfg: GazeT5ForCausalLMConfig) -> Dict[str, torch.Tensor | List[List[str]]]:
    """
    Collates a batch of data points from ArticleGazeDS.
    Handles tokenization, padding, and structuring data for GazeT5ForCausalLM.

    Args:
        batch: A list of dictionaries, where each dict is an output of ArticleGazeDS.__getitem__.
        tok: The pre-trained tokenizer.
        cfg: The main GazeT5ForCausalLMConfig instance.

    Returns:
        A dictionary containing tensors ready for the model's forward pass:
        - input_ids: Padded prompt token IDs.
        - attention_mask: Padded prompt attention mask.
        - labels: Padded target (summary) token IDs.
        - gaze_num: Padded gaze numeric data tensor [B, S, Lg, F].
        - gaze_mask: Padded gaze boolean mask tensor [B, S, Lg].
        - sentences: List of lists containing sentence strings for the batch [B][S].
                     (Note: sentences is not a tensor).
    """
    # Use CPU for collation steps, move to GPU later if needed (usually handled by Trainer/DataLoader)
    # device = torch.device("cpu") # Tensors created will be on CPU

    # ───── 1. Tokenize and Pad Prompts ───────────────────────────────
    prompt_texts = [item["prompt_text"] for item in batch]

    # Prepare prompts using the chat template (adjust roles/content as needed)
    # This assumes the base LLM works well with this structure.
    prompt_ids_list = []
    for text in prompt_texts:
        msgs = [
            # {"role": "system", "content": "You are a helpful assistant."}, # Optional system prompt
            {"role": "user", "content": text}
        ]
        # Tokenize for training: NO generation prompt added here.
        # Handle potential token type IDs if the model requires them.
        ids = tok.apply_chat_template(
                msgs,
                add_generation_prompt=False, # Important for training
                return_tensors="pt",
                # Consider adding truncation if prompts can exceed model max length
                # max_length=tok.model_max_length,
                # truncation=True,
                ).squeeze(0) # Remove batch dim added by return_tensors
        prompt_ids_list.append(ids)

    # Pad the tokenized prompts to the longest prompt in the batch.
    # `padding=True` uses the tokenizer's pad token ID.
    # `padding_side` should match tokenizer's setting (usually 'left' for decoder models).
    prompt_padding_output = tok.pad(
        {"input_ids": prompt_ids_list},
        padding=True,
        return_tensors="pt",
        return_attention_mask=True # Ensure attention mask is generated
    )
    input_ids = prompt_padding_output["input_ids"]          # Shape: [B, L_prompt_max]
    attention_mask = prompt_padding_output["attention_mask"] # Shape: [B, L_prompt_max]


    # ───── 2. Tokenize and Pad Labels (Summaries) ────────────────────
    summary_texts = [item["summary_text"] for item in batch]

    # Tokenize the target summaries. Add EOS token for clear sequence end signal.
    # Do *not* add BOS token here usually, as decoder models handle that.
    # `add_special_tokens=False` prevents tokenizer from adding BOS again if already handled.
    summary_ids_list = [
        tok.encode(text + tok.eos_token, add_special_tokens=False)
        for text in summary_texts
    ]

    # Pad the tokenized labels. The model's forward pass will later ignore
    # loss for pad tokens and prompt tokens by using -100 label padding internally.
    # Here, we just pad with the standard pad token ID.
    label_padding_output = tok.pad(
        {"input_ids": summary_ids_list},
        padding=True,
        return_tensors="pt"
        # No attention mask needed for labels during standard training loss calculation
    )
    # This tensor contains the padded target token IDs.
    labels = label_padding_output["input_ids"]              # Shape: [B, L_labels_max]


    # ───── 3. Pad Gaze Data and Sentences ────────────────────────────
    # Determine max number of sentences and max gaze sequence length in the batch.
    max_sents_in_batch = max(len(item["sentences_list"]) for item in batch)

    max_gaze_len_in_batch = 0
    for item in batch:
        # Check lengths only for sentences that actually have gaze data
        valid_gaze_lengths = [len(t) for t in item["gaze_nums_list"] if hasattr(t, '__len__') and len(t) > 0]
        if valid_gaze_lengths:
            max_gaze_len_in_batch = max(max_gaze_len_in_batch, max(valid_gaze_lengths))

    B = len(batch) # Batch size
    # Get the expected number of numeric gaze features from config
    eye_cfg = cfg.get_eye_t5_config()
    in_dim_numeric = eye_cfg.in_dim_numeric

    # Initialize tensors for padded gaze numbers and masks with zeros.
    gaze_num_padded  = torch.zeros(B, max_sents_in_batch, max_gaze_len_in_batch, in_dim_numeric)
    gaze_mask_padded = torch.zeros(B, max_sents_in_batch, max_gaze_len_in_batch, dtype=torch.bool)

    # Prepare the list of lists for sentences (padding inner lists with empty strings).
    sentences_padded_list = []

    # Iterate through the batch to fill the padded gaze tensors and sentence list.
    for i, item in enumerate(batch):
        sents = item["sentences_list"]
        gaze_nums = item["gaze_nums_list"]
        gaze_masks = item["gaze_masks_list"]
        num_sents_in_item = len(sents)

        # Pad the list of sentences for this item with empty strings if needed.
        sentence_padding = [""] * (max_sents_in_batch - num_sents_in_item)
        sentences_padded_list.append(sents + sentence_padding)

        # Copy gaze data into the padded tensors.
        for j in range(num_sents_in_item):
            x_num = gaze_nums[j]
            x_mask = gaze_masks[j]
            # Check if it's a tensor and has data before processing
            if isinstance(x_num, torch.Tensor):
                L = len(x_num)
                if L > 0:
                    # Double-check feature dimension before assignment
                    if x_num.shape[1] == in_dim_numeric:
                        gaze_num_padded[i, j, :L]  = x_num
                        gaze_mask_padded[i, j, :L] = x_mask
                    else:
                        # Log a warning if dimensions don't match expected config
                        print(f"Warning: Gaze feature dim mismatch in collate_fn for "
                              f"batch item {i}, sentence {j}. Expected {in_dim_numeric}, "
                              f"got {x_num.shape[1]}. Skipping this gaze sequence.")
            elif len(x_num) > 0 : # Handle potential non-tensor but non-empty data if loading was odd
                 print(f"Warning: Gaze data is not a tensor for batch item {i}, sentence {j}. Type: {type(x_num)}. Skipping.")


    # ───── 4. Construct Final Batch Dictionary ───────────────────────
    # Ensure keys match what the model's forward method expects.
    final_batch = {
        "input_ids":      input_ids,          # Padded prompt token IDs [B, Lp]
        "attention_mask": attention_mask,     # Padded prompt attention mask [B, Lp]
        "labels":         labels,             # Padded target token IDs [B, Lr]
        "gaze_num":       gaze_num_padded,    # Padded gaze numeric data [B, S, Lg, F]
        "gaze_mask":      gaze_mask_padded,   # Padded gaze boolean mask [B, S, Lg]
        "sentences":      sentences_padded_list, # List[List[str]], batch B, inner list size S
    }

    return final_batch