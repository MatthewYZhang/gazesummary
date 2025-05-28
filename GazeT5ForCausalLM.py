# GazeT5ForCausalLM.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import List, Optional, Tuple, Dict, Any

# Assuming EyeAutoEncoderT5 is correctly imported
from gazeT5 import EyeAutoEncoderT5
# Import the *new* config classes
from config import GazeT5ForCausalLMConfig, EyeT5Config # Import both

class GazeT5ForCausalLM(PreTrainedModel):
    config_class = GazeT5ForCausalLMConfig  # Link to the main config class

    def __init__(self, config: GazeT5ForCausalLMConfig):
        super().__init__(config)
        self.config = config # Store main config

        # --- Instantiate Gaze Encoder ---
        # Get the specific EyeT5Config object from the main config
        eye_t5_cfg: EyeT5Config = config.get_eye_t5_config()
        # Pass the config object directly to EyeAutoEncoderT5
        self.gaze_enc = EyeAutoEncoderT5(eye_t5_cfg)

        # --- Load Gaze Encoder Pretrained Weights (Stage 1) ---
        try:
            print(f"Attempting to load Gaze Encoder weights from: {config.gaze_ckpt_path}")
            # Ensure gaze_ckpt_path exists and is accessible
            state = torch.load(config.gaze_ckpt_path, map_location="cpu")["model_state"]
            # Load state dict. Strict=False might be needed if proj layer size changed.
            missing_keys, unexpected_keys = self.gaze_enc.load_state_dict(state, strict=False)
            if missing_keys: print(f"Warning: Missing keys in gaze_enc state_dict: {missing_keys}")
            if unexpected_keys: print(f"Warning: Unexpected keys in gaze_enc state_dict: {unexpected_keys}")
            print("Successfully loaded Gaze Encoder weights.")
        except FileNotFoundError:
            print(f"Warning: Gaze checkpoint not found at {config.gaze_ckpt_path}. Initializing gaze encoder from scratch.")
        except Exception as e:
            print(f"Warning: Error loading Gaze checkpoint {config.gaze_ckpt_path}: {e}. Initializing gaze encoder from scratch.")

        # Freeze gaze encoder except pool.query & proj
        for p in self.gaze_enc.parameters(): p.requires_grad = False
        # Ensure these components exist before setting requires_grad
        if hasattr(self.gaze_enc, 'encoder') and hasattr(self.gaze_enc.encoder, 'pool'):
             self.gaze_enc.encoder.pool.query.requires_grad_(True)
        else:
             print("Warning: gaze_enc.encoder.pool.query not found for unfreezing.")
        if hasattr(self.gaze_enc, 'proj'):
             for p in self.gaze_enc.proj.parameters(): p.requires_grad = True
        else:
             print("Warning: gaze_enc.proj not found for unfreezing.")


        # --- Load Base LLM ---
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_name,
            # Add other necessary args like torch_dtype, device_map if needed
            # torch_dtype=torch.bfloat16,
            trust_remote_code=True # Often needed for Qwen
        )
        # Freeze the base LLM initially
        for p in self.llm.parameters(): p.requires_grad = False

        # Store LLM's hidden dimension
        self.llm_hidden_size = self.llm.config.hidden_size

        # --- Verify Projection Dimension Match ---
        # Check if the gaze encoder's projection output dim matches LLM embedding dim
        if hasattr(self.gaze_enc, 'proj'):
            gaze_proj_output_dim = self.gaze_enc.proj.out_features
            if gaze_proj_output_dim != self.llm_hidden_size:
                 print(f"Warning: Gaze encoder projection output dimension ({gaze_proj_output_dim}) "
                       f"does not match LLM hidden size ({self.llm_hidden_size}). "
                       f"Ensure EyeT5Config.proj_dim ({eye_t5_cfg.proj_dim}) is set correctly "
                       f"and matches the expected LLM dimension.")
                 # If this mismatch is critical, you might raise an error or attempt to resize/reinitialize
                 # raise ValueError("Dimension mismatch between gaze projection and LLM embeddings.")
            else:
                 print(f"Gaze projection dimension ({gaze_proj_output_dim}) matches LLM hidden size ({self.llm_hidden_size}).")
        else:
            print("Warning: Cannot verify gaze projection dimension as gaze_enc.proj layer not found.")


    # No need for _embed_normal_tokens, use llm's embedding layer directly
    def get_input_embeddings(self) -> nn.Module:
         return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module):
         self.llm.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,             # [B, L_prompt] (e.g., instruction+article)
        attention_mask: Optional[torch.LongTensor] = None,        # [B, L_prompt] (mask for input_ids)
        gaze_num: Optional[torch.FloatTensor] = None,             # [B, S, Lg, F]
        gaze_mask: Optional[torch.BoolTensor] = None,             # [B, S, Lg]
        sentences: Optional[List[List[str]]] = None,              # len B
        labels: Optional[torch.LongTensor] = None,                # [B, L_summary] (target summary token IDs)
        inputs_embeds: Optional[torch.FloatTensor] = None,        # [B, L, D] (Alternative input, usually for generation)
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass for GazeT5ForCausalLM.

        During Training (labels are provided):
            - Concatenates gaze_embeds, prompt_embeds, and summary_embeds.
            - Creates labels padded with -100 for gaze and prompt sections.
            - Passes the concatenated embeddings and padded labels to the LLM.
        During Inference (labels are None):
            - Concatenates gaze_embeds and prompt_embeds.
            - Passes these embeddings to the LLM.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None and input_ids is not None:
            raise ValueError("Cannot provide both `input_ids` and `inputs_embeds`.")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("Either `input_ids` or `inputs_embeds` must be provided.")

        # If inputs_embeds are provided directly (likely during generation with past_key_values),
        # we skip the gaze/prompt embedding calculation here.
        if inputs_embeds is not None:
            # When using generate() with past_key_values, inputs_embeds might be just the next token's embedding.
            # The model call below handles this case directly. Labels are typically None here.
            if labels is not None:
                print("Warning: Labels provided directly with inputs_embeds. Ensure sequence lengths match if this is intended for training.")
            # Pass directly to LLM - assumes labels (if provided) match inputs_embeds length
            return self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask, # Must correspond to inputs_embeds
                labels=labels,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                return_dict=return_dict,
            )

        # --- Standard case: Start from input_ids, gaze_num, sentences, etc. ---
        if gaze_num is None or gaze_mask is None or sentences is None:
            raise ValueError("Gaze information (gaze_num, gaze_mask, sentences) required when using input_ids.")

        device = input_ids.device
        B = input_ids.shape[0]
        S, Lg, F = gaze_num.shape[1], gaze_num.shape[2], gaze_num.shape[3]

        # 1. Gaze Embeddings
        flat_num   = gaze_num.reshape(B*S, Lg, F)
        flat_mask  = gaze_mask.reshape(B*S, Lg)
        flat_sents = sum(sentences, [])
        k_tokens = self.config.get_eye_t5_config().k_tokens
        _, flat_tok  = self.gaze_enc(flat_num, flat_mask, flat_sents)
        gaze_tok     = flat_tok.reshape(B, S * k_tokens, -1) # [B, L_gaze, D]
        gaze_att     = torch.ones(B, S * k_tokens, dtype=torch.long, device=device)
        L_gaze = gaze_tok.shape[1]

        # 2. Prompt Embeddings (from input_ids)
        prompt_emb = self.llm.get_input_embeddings()(input_ids) # [B, L_prompt, D]
        prompt_att = attention_mask # Use the attention mask passed in for the prompt [B, L_prompt]
        if prompt_att is None: # Ensure mask exists if input_ids are given
            prompt_att = torch.ones_like(input_ids)
        L_prompt = prompt_emb.shape[1]


        # --- Branch based on whether it's training (labels provided) or inference ---

        if labels is not None:
            # --- TRAINING PATH ---
            # 3a. Summary Embeddings (from labels)
            summary_emb = self.llm.get_input_embeddings()(labels) # [B, L_summary, D]
            summary_att = torch.ones_like(labels)                # [B, L_summary]
            L_summary = summary_emb.shape[1]

            # 4a. Concatenate ALL embeddings for input
            final_inputs_embeds = torch.cat([gaze_tok, prompt_emb, summary_emb], dim=1) # [B, L_gaze+L_prompt+L_summary, D]

            # 5a. Concatenate ALL attention masks
            final_attention_mask = torch.cat([gaze_att, prompt_att, summary_att], dim=1) # [B, L_gaze+L_prompt+L_summary]

            # 6a. Create padded labels matching the full input sequence length
            pad_gaze_prompt_len = L_gaze + L_prompt
            pad_gaze_prompt = torch.full((B, pad_gaze_prompt_len), -100, dtype=torch.long, device=device)
            # Concatenate padding with the actual summary labels
            final_labels = torch.cat([pad_gaze_prompt, labels], dim=1) # [B, L_gaze+L_prompt+L_summary]

            # 7a. Call LLM with combined inputs and correctly padded labels
            # print(f"Training Shapes-> Embeds: {final_inputs_embeds.shape}, Mask: {final_attention_mask.shape}, Labels: {final_labels.shape}")
            outputs = self.llm(
                inputs_embeds=final_inputs_embeds,
                attention_mask=final_attention_mask,
                labels=final_labels, # Lengths now match
                past_key_values=past_key_values, # Usually None during training unless using grad checkpointing specific techniques
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache if use_cache is not None else False, # Typically False during training unless needed
                return_dict=return_dict,
            )

        else:
            # --- INFERENCE PATH ---
            # 3b. Concatenate only gaze and prompt embeddings
            final_inputs_embeds = torch.cat([gaze_tok, prompt_emb], dim=1) # [B, L_gaze+L_prompt, D]

            # 4b. Concatenate only gaze and prompt attention masks
            final_attention_mask = torch.cat([gaze_att, prompt_att], dim=1) # [B, L_gaze+L_prompt]

            # 5b. Call LLM without labels
            # print(f"Inference Shapes-> Embeds: {final_inputs_embeds.shape}, Mask: {final_attention_mask.shape}")
            outputs = self.llm(
                inputs_embeds=final_inputs_embeds,
                attention_mask=final_attention_mask,
                labels=None, # No labels provided
                past_key_values=past_key_values, # Used during generation
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache if use_cache is not None else True, # Typically True during generation
                return_dict=return_dict,
            )

        return outputs


    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: Optional[bool] = None,
        # Custom arguments needed for gaze embedding must be passed here!
        gaze_num: Optional[torch.FloatTensor] = None,
        gaze_mask: Optional[torch.BoolTensor] = None,
        sentences: Optional[List[List[str]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # ... (Handle past_key_values and direct inputs_embeds cases - remain the same) ...

        # --- Initial step or when inputs_embeds are not provided ---
        if not past_key_values and inputs_embeds is None:
             if gaze_num is None or gaze_mask is None or sentences is None:
                 raise ValueError("Gaze information (gaze_num, gaze_mask, sentences) must be passed "
                                  "to model.generate() for the initial step.")

             device = input_ids.device
             B = input_ids.shape[0]
             S, Lg, F = gaze_num.shape[1], gaze_num.shape[2], gaze_num.shape[3]

             flat_num   = gaze_num.reshape(B*S, Lg, F)
             flat_mask  = gaze_mask.reshape(B*S, Lg)
             flat_sents = sum(sentences, [])

             # Get k_tokens from the specific EyeT5Config instance
             k_tokens = self.config.get_eye_t5_config().k_tokens

             _, flat_tok  = self.gaze_enc(flat_num, flat_mask, flat_sents)
             gaze_tok     = flat_tok.reshape(B, S * k_tokens, -1) # Use k_tokens
             gaze_att     = torch.ones(B, S * k_tokens, dtype=torch.long, device=device)
             L_gaze = gaze_tok.shape[1]

             # Embed prompt tokens
             prompt_emb = self.llm.get_input_embeddings()(input_ids)

             # Concatenate
             inputs_embeds = torch.cat([gaze_tok, prompt_emb], dim=1)

             # Create combined attention mask
             if attention_mask is None:
                 attention_mask = torch.ones_like(input_ids)
             full_attention_mask = torch.cat([gaze_att, attention_mask], dim=1)

        elif past_key_values:
             # We already have past, only need embeds for the new token(s)
             inputs_embeds = self.llm.get_input_embeddings()(input_ids)
             full_attention_mask = attention_mask
        # else: inputs_embeds was provided directly, handled earlier


        model_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": full_attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            **kwargs,
        }
        return model_inputs

    # _reorder_cache remains the same
    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm._reorder_cache(past_key_values, beam_idx)