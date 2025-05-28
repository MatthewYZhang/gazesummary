# stage1_train.py
import os, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from config import EyeT5Config, asdict
from gazeT5 import EyeAutoEncoderT5
from stage1_dataset import GazeSentenceDS, collate


# ── main ────────────────────────────────────────────────────────────
def train(cfg: EyeT5Config):
    device  = "cuda:0" if torch.cuda.is_available() else "cpu"
    model   = EyeAutoEncoderT5(cfg).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=cfg.lr_s1, weight_decay=cfg.weight_decay_s1)
    mse     = nn.MSELoss()

    # tiny sentence‑embedder (not used by auto‑encoder yet, but computed & saved)
    sent_emb = SentenceTransformer(cfg.sent_model_name, device=device)
    loader   = DataLoader(GazeSentenceDS("data/gaze_stage1", "train", cfg), batch_size=128, shuffle=True,
                          collate_fn=collate, pin_memory=True)

    # print total parameter size of model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total_params:,}")
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    step = 0
    while step < cfg.max_steps_s1:
        for x_num, mask, sents, lens in loader:
            x_num, mask = x_num.to(device), mask.to(device)

            recon, gaze_tokens = model(x_num, mask, sents)   # 传入 sentences
            loss = mse(recon, x_num)

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_s1)
            opt.step()

            if step % 5 == 0:
                print(f"step {step:6d}  loss {loss.item():.4f}")

            step += 1
            if step >= cfg.max_steps_s1: break
        # break

    # ── save checkpoints ──────────────────────────────────────────
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.save({"cfg": asdict(cfg),
                "model_state": model.state_dict()},
               os.path.join(cfg.save_dir, "gaze_t5_model.pt"))
    # model.encoder.encoder.save_pretrained(os.path.join(cfg.save_dir, "t5_encoder"))
    # sent_emb.save(os.path.join(cfg.save_dir, "sent_embedder"))
    print(f"saved to {cfg.save_dir}")

if __name__ == "__main__":
    cfg = EyeT5Config()
    train(cfg)
