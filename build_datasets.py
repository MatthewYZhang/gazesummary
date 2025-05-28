# build_datasets.py
"""
Creates two HF datasets:
  data/gaze_stage1  – sentence-level rows   (for the auto-encoder)
  data/gaze_stage23 – article-level rows   (for alignment + joint fine-tune)
"""
import json, random, pathlib
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from collections import defaultdict

SRC_ROOT = Path("data")
OUT1 = "data/gaze_stage1"
OUT2 = "data/gaze_stage23"
VAL_PCT = TEST_PCT = 0.10

sent_rows = []                          # for Stage-1
article_rows = []                       # for Stage-2/3

screen_width = 1920
screen_height = 1080
metric = 0.001 # convert to seconds rather than milliseconds

for split in ["raw_split1", "raw_split2"]:
    for folder in tqdm(list((SRC_ROOT / split).glob("*"))):
        if not folder.is_dir(): continue
        j_path, t_path = folder/"training_data.json", folder/"heatmap_summary.txt"
        if not (j_path.exists() and t_path.exists()): continue

        with open(j_path) as jf, open(t_path) as tf:
            sentence_dict = json.load(jf)      # {"0": {...}, …}
            summary_text  = tf.read().strip()
        source = f"{split}/{folder.name}"
        article_sentence_ids = []
        for sid, s in sentence_dict.items():
            sentence_txt = " ".join(s["words"])
            if len(s['gaze_x']) != len(s['gaze_y']) or len(s['gaze_x']) != len(s['gaze_dur']) or len(s['gaze_x']) == 0:
                # print(f"Warning: sentence {sid} in {source} has inconsistent gaze data")
                continue
            row = {
                # "gaze_x":   s["gaze_x"],
                # "gaze_y":   s["gaze_y"],
                # "gaze_dur": s["gaze_dur"],
                "gaze_x":   [x/screen_width for x in s["gaze_x"]],
                "gaze_y":   [y/screen_height for y in s["gaze_y"]],
                "gaze_dur": [d*metric for d in s["gaze_dur"]],
                "sentence": sentence_txt,
                "article_id": source       # numeric folder name
            }
            sent_rows.append(row)
            article_sentence_ids.append(len(sent_rows)-1)   # index in sent_rows
        article_rows.append({
            "sentence_indices": article_sentence_ids,       # idx list into Stage-1 set
            "summary": summary_text,
            "article_id": source
        })

VAL_PCT  = 0.10
TEST_PCT = 0.10
OUT1 = "data/gaze_stage1"   # sentence-level dataset
OUT2 = "data/gaze_stage23"  # article-level dataset

random.seed(42)

random.shuffle(article_rows)
n_art = len(article_rows)
n_val_art  = int(n_art * VAL_PCT)
n_test_art = int(n_art * TEST_PCT)

art_val   = article_rows[:n_val_art]
art_test  = article_rows[n_val_art:n_val_art+n_test_art]
art_train = article_rows[n_val_art+n_test_art:]

split_of_article = {}
for r in art_train: split_of_article[r["article_id"]] = "train"
for r in art_val:   split_of_article[r["article_id"]] = "val"
for r in art_test:  split_of_article[r["article_id"]] = "test"

# ------------------------------------------------------------------
# 2) build sentence splits *with* global→local index remapping
# ------------------------------------------------------------------
sent_splits = {"train": [], "val": [], "test": []}
index_map   = {"train": {}, "val": {}, "test": {}}   # global→local

for global_i, s_row in enumerate(sent_rows):
    split = split_of_article[s_row["article_id"]]
    local_i = len(sent_splits[split])
    index_map[split][global_i] = local_i
    sent_splits[split].append(s_row)

# ------------------------------------------------------------------
# 3) rewrite each article‘s sentence_indices to split-local
# ------------------------------------------------------------------
def remap_indices(article_list, split_name):
    m = index_map[split_name]
    for art in article_list:
        art["sentence_indices"] = [m[gi] for gi in art["sentence_indices"]]

remap_indices(art_train, "train")
remap_indices(art_val,   "val")
remap_indices(art_test,  "test")

# ------------------------------------------------------------------
# 4) save both datasets
# ------------------------------------------------------------------
Path(OUT1).mkdir(parents=True, exist_ok=True)
DatasetDict({
    "train": Dataset.from_list(sent_splits["train"]),
    "val":   Dataset.from_list(sent_splits["val"]),
    "test":  Dataset.from_list(sent_splits["test"]),
}).save_to_disk(OUT1)

Path(OUT2).mkdir(parents=True, exist_ok=True)
DatasetDict({
    "train": Dataset.from_list(art_train),
    "val":   Dataset.from_list(art_val),
    "test":  Dataset.from_list(art_test),
}).save_to_disk(OUT2)

print("✓ both datasets saved with consistent indices")

# ------------------------------------------------------------------
# 6)  Write a human-readable preview file
# ------------------------------------------------------------------
def count_sents(article_list):
    tot = 0
    for art in article_list:
        tot += len(art["sentence_indices"])
    return tot

preview_lines = []

for split_name, art_list in [("TRAIN", art_train),
                             ("VAL",   art_val),
                             ("TEST",  art_test)]:
    n_art  = len(art_list)
    n_sent = count_sents(art_list)
    preview_lines.append(f"== {split_name}  ({n_art} articles, {n_sent} sentences) ==\n")
    for art in art_list:
        preview_lines.append(f"article_id: {art['article_id']:>6}   "
                             f"sentences: {len(art['sentence_indices'])}\n")
    preview_lines.append("\n")

with open(Path(OUT2, "preview.txt"), "w") as f:      # save next to article dataset
    f.writelines(preview_lines)

print("✓ Human-readable preview written to", Path(OUT2, "preview.txt"))

# ------------------------------------------------------------------
# 6)  Human-readable previews for BOTH datasets
# ------------------------------------------------------------------
from textwrap import shorten

def count_sents(article_list):
    return sum(len(a["sentence_indices"]) for a in article_list)

# --- article-level preview ---------------------------------------
preview_art = []
for split_name, art_list in [("TRAIN", art_train),
                             ("VAL",   art_val),
                             ("TEST",  art_test)]:
    n_art  = len(art_list)
    n_sent = count_sents(art_list)
    preview_art.append(f"== {split_name}  ({n_art} articles, {n_sent} sentences) ==\n")
    for art in art_list:
        preview_art.append(f"article_id: {art['article_id']:>6}   "
                           f"sentences: {len(art['sentence_indices'])}\n")
    preview_art.append("\n")

Path(OUT2, "preview.txt").write_text("".join(preview_art))
print("✓ Article-level preview →", Path(OUT2, "preview.txt"))

# --- sentence-level preview --------------------------------------
preview_sent = []
for split_name, rows in [("TRAIN", sent_splits["train"]),
                         ("VAL",   sent_splits["val"]),
                         ("TEST",  sent_splits["test"])]:
    preview_sent.append(f"== {split_name}  ({len(rows)} sentences) ==\n")
    for idx, r in enumerate(rows):         # cap to first 300 for brevity
        snippet = shorten(r["sentence"], width=70, placeholder="…")
        preview_sent.append(f"[{idx:05}] article:{r['article_id']:>6}  {snippet}\n")
    preview_sent.append("\n")

Path(OUT1, "preview.txt").write_text("".join(preview_sent))
print("✓ Sentence-level preview →", Path(OUT1, "preview.txt"))
