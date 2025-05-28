# modules_t5.py
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from config import EyeT5Config
from typing import List
from sentence_transformers import SentenceTransformer

# ─────────── attentive K‑token pooling ─────────────────────────────
class AttentionPooling(nn.Module):
    def __init__(self, dm, k):
        super().__init__()
        self.query = nn.Parameter(torch.randn(k, dm))

    def forward(self, hidden, mask):
        # hidden: [B,T,dm]   mask: [B,T]  (True=keep)
        logits = torch.matmul(self.query, hidden.transpose(1, 2))    # [B,K,T]
        logits = logits.masked_fill(~mask[:, None, :], -1e4)
        attn   = F.softmax(logits, dim=-1)                           # [B,K,T]
        return torch.matmul(attn, hidden)                            # [B,K,dm]

# ─────────── encoder : numeric gaze → T5Stack → K embeddings ───────
class EyeEncoderT5(nn.Module):
    def __init__(self, cfg: EyeT5Config):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.in_dim_numeric, cfg.d_model)

        enc_cfg = T5Config(
            vocab_size           = 1,                 # dummy – inputs_embeds is used
            d_model              = cfg.d_model,
            d_ff                 = cfg.d_model * 4,
            num_layers           = cfg.n_encoder_layers,
            num_heads            = cfg.n_heads,
            dropout_rate         = cfg.dropout,
            is_decoder           = False,
            is_encoder_decoder   = False,
            use_cache            = False,
        )
        # dummy embedding because T5Stack expects one – never used at run‑time
        dummy_embed = nn.Embedding(1, cfg.d_model)
        self.encoder = T5Stack(enc_cfg, embed_tokens=dummy_embed)
        self.pool    = AttentionPooling(cfg.d_model, cfg.k_tokens)

    def forward(self, x_num, mask):
        x_emb = self.in_proj(x_num)                                   # [B,T,dm]
        enc_out = self.encoder(inputs_embeds=x_emb,
                               attention_mask=mask).last_hidden_state
        return self.pool(enc_out, mask)                               # [B,K,dm]

# ─────────── decoder : K embeddings → reconstruct gaze stream ──────
class EyeDecoderT5(nn.Module):
    def __init__(self, cfg: EyeT5Config):
        super().__init__()
        dec_cfg = T5Config(
            vocab_size           = 1,
            d_model              = cfg.d_model,
            d_ff                 = cfg.d_model * 4,
            num_layers           = cfg.n_decoder_layers,
            num_heads            = cfg.n_heads,
            dropout_rate         = cfg.dropout,
            is_decoder           = True,
            is_encoder_decoder   = False,
            use_cache            = False,
        )
        self.decoder = T5Stack(dec_cfg, embed_tokens=nn.Embedding(1, cfg.d_model))
        self.out     = nn.Linear(cfg.d_model, cfg.in_dim_numeric)

    def forward(self, summary, tgt_len):
        # summary: [B,K,dm]
        B, K, dm = summary.size()
        # create an all‑zero "decoder prompt" (will be cross‑attended)
        dec_in  = torch.zeros(B, tgt_len, dm, device=summary.device)
        dec_out = self.decoder(inputs_embeds       = dec_in,
                               encoder_hidden_states = summary,
                               encoder_attention_mask = torch.ones(B, K, dtype=torch.long,
                                                                    device=summary.device)
                               ).last_hidden_state
        return self.out(dec_out)                                      # [B,T,F_num]

# ─────────── full auto‑encoder + projection ─────────────────────────
class EyeAutoEncoderT5(nn.Module):
    """
    • encoder  : numeric gaze  →  K×dm
    • sent_emb : MiniLM(sentence)  →  se_dim
    • concat   : [B,K,dm+se_dim]  → proj → [B,K,proj_dim]  (gaze‑tokens)
    • decoder  : K×dm  →  recon gaze stream
    """
    def __init__(self, cfg: EyeT5Config):
        super().__init__()
        self.cfg      = cfg
        self.encoder  = EyeEncoderT5(cfg)
        # ── MiniLM sentence embedder (freeze) ───────────────────────
        self.sent_model = SentenceTransformer(cfg.sent_model_name)
        for p in self.sent_model.parameters():
            p.requires_grad = False
        self.se_dim   = self.sent_model.get_sentence_embedding_dimension()

        self.proj     = nn.Linear(cfg.d_model + self.se_dim, cfg.proj_dim)
        self.decoder  = EyeDecoderT5(cfg)

    def forward(self,
                x_num        : torch.Tensor,      # [B,T,F_num]
                mask         : torch.Tensor,      # [B,T] bool
                sentences    : List[str]          # len B
               ):
        print(x_num.shape, mask.shape, len(sentences))
        summary = self.encoder(x_num, mask)                    # [B,K,dm]

        # ── sentence embedding (no grad) ───────────────────────────
        
        # print(len(sentences), type(sentences), type(sentences[0]))
        with torch.no_grad():
            sent_emb = self.sent_model.encode(
                sentences, convert_to_tensor=True,
                device=x_num.device, normalize_embeddings=False
            )                                                  # [B,se_dim]
        # change the embedding to 0 if the the item in sentences is ""
        # use the item in sentences to decide!

        mask_flat = torch.tensor(
            [s != "" for s in sentences],
            device=x_num.device, dtype=torch.bool
        )
        sent_emb = sent_emb.masked_fill(mask_flat[:, None] == 0, 0)  # [B,se_dim]
        # broadcast to K tokens, concat then project
        # print(sent_emb.shape)
        sent_exp = sent_emb.unsqueeze(1).expand(-1, self.cfg.k_tokens, -1)
        # print(summary.shape, sent_exp.shape)
        concat   = torch.cat([summary, sent_exp], dim=-1)      # [B,K,dm+se]
        tokens   = self.proj(concat)                           # [B,K,proj_dim]

        recon    = self.decoder(summary, x_num.size(1))        # [B,T,F_num]
        return recon, tokens
