# config.py
import json
from dataclasses import dataclass, asdict, field # Import field for potential defaults
from typing import Optional, Dict, Any
from transformers import PretrainedConfig

# User's detailed EyeT5Config using dataclass
@dataclass
class EyeT5Config:
    # ────────── raw gaze stream ───────────────────────────────────────
    in_dim_full:     int = 4     # total features *with* the 1‑D “sentence‑id”
    text_dim:        int = 1
    # Calculate in_dim_numeric based on others
    in_dim_numeric:  int = in_dim_full - text_dim # Placeholder, will be set in __post_init__

    # ────────── freshly‑initialised T5 stack ─────────────────────────
    d_model:         int = 512        # same as t5‑small, but **random init**
    n_encoder_layers:int = 6
    n_decoder_layers:int = 2
    n_heads:         int = 8
    dropout:         float = 0.1

    # ────────── gaze→LLM tokens ─────────────────────────────────────
    k_tokens:        int = 8
    proj_dim:        int = 2048       # Should match LLM embedding size (e.g., Qwen2.5‑3B = 2048)

    # ────────── optimisation for stage 1 ────────────────────────────────────────
    lr_s1:              float = 1e-3
    weight_decay_s1:    float = 1e-2
    max_steps_s1:       int = 2000
    grad_clip_s1:       float = 1.0

    # ────────── optimisation for stage 2 ────────────────────────────────────────
    lr_s2:              float = 1e-3
    weight_decay_s2:    float = 1e-2
    # max_steps_s2:       int = 10000 # Example value if needed later
    grad_clip:          float = 1.0 # Assuming this is grad_clip_s2

    # ────────── sentence embedding model (MiniLM) ───────────────────
    sent_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ────────── checkpoint path (Note: Path for Stage 1 model is better stored in GazeT5ForCausalLMConfig)──
    # save_dir:        str = "./trained_models" # We might not need this here if GazeT5ForCausalLMConfig handles paths

    # def __post_init__(self):
    #     # Automatically calculate in_dim_numeric
    #     self.in_dim_numeric = self.in_dim_full - self.text_dim

    # helper to dump whole cfg when saving
    def to_dict(self): return asdict(self)

    # Optional: Add a from_dict if needed elsewhere, but direct init works too
    # @classmethod
    # def from_dict(cls, config_dict):
    #     return cls(**config_dict)


# Updated Config for the main model, inheriting from Hugging Face's PretrainedConfig
class GazeT5ForCausalLMConfig(PretrainedConfig):
    model_type = "gaze_t5_for_causal_lm" # Necessary identifier

    def __init__(
        self,
        llm_name="Qwen/Qwen2.5-3B-Instruct", # Base LLM identifier
        gaze_ckpt_path="./trained_models/gaze_t5_model.pt", # Path to Stage 1 weights
        eye_t5_config_overrides: Optional[Dict[str, Any]] = None, # Allow overriding EyeT5 defaults
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_name = llm_name
        self.gaze_ckpt_path = gaze_ckpt_path

        # Create default EyeT5Config and apply overrides
        default_eye_t5_params = asdict(EyeT5Config())
        if eye_t5_config_overrides:
            default_eye_t5_params.update(eye_t5_config_overrides)

        # Store the potentially overridden EyeT5Config parameters as a dictionary
        # This dictionary gets saved in the main config.json
        self.eye_t5_config_dict = default_eye_t5_params

    # Helper to get a reconstructed EyeT5Config object easily
    def get_eye_t5_config(self) -> EyeT5Config:
        """Instantiates EyeT5Config from the stored dictionary."""
        return EyeT5Config(**self.eye_t5_config_dict)