# build_test_dataset.py
"""
Creates two HF datasets from user_real_data:
  user_real_data/stage1  – sentence-level rows (test split only)
  user_real_data/stage23 – article-level rows (test split only, empty summaries)

Processes subdirectories in user_real_data matching the pattern:
  - Starts with two or three letters
  - Followed by a digit
  - Followed by an underscore
  - Example: djx1_some_identifier, abc2_another_one
"""
import json
import re # Import regular expressions
import pathlib
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from collections import defaultdict
from textwrap import shorten

# --- Configuration ---
SRC_ROOT = Path("user_real_data")    # Input directory containing article subfolders
OUT1 = SRC_ROOT / "stage1"         # Output for sentence-level dataset
OUT2 = SRC_ROOT / "stage23"        # Output for article-level dataset
FOLDER_PATTERN = r'^[a-zA-Z]{2,3}\d_.*' # Regex pattern for valid folder names

# Screen dimensions and metric for normalization/conversion (adjust if needed)
screen_width = 1920
screen_height = 1080
metric = 0.001 # convert milliseconds to seconds

# --- Data Collection ---
sent_rows = []                          # For Stage-1 (sentences)
article_rows = []                       # For Stage-2/3 (articles)

print(f"Scanning '{SRC_ROOT}' for valid article folders...")

# Ensure source directory exists
if not SRC_ROOT.is_dir():
    print(f"Error: Source directory '{SRC_ROOT}' not found.")
    exit()

# Iterate through potential article folders
for folder in tqdm(list(SRC_ROOT.glob("*"))):
    if not folder.is_dir():
        # print(f"Skipping non-directory: {folder.name}")
        continue

    # Check if folder name matches the required pattern
    if not re.match(FOLDER_PATTERN, folder.name):
        # print(f"Skipping folder with invalid name format: {folder.name}")
        continue

    j_path = folder / "training_data.json"
    if not j_path.exists():
        print(f"Warning: Skipping folder '{folder.name}' - missing 'training_data.json'")
        continue

    # Load sentence data
    try:
        with open(j_path) as jf:
            sentence_dict = json.load(jf) # {"0": {...}, ...} or {"sentence_0":{...}, ...} etc.
    except json.JSONDecodeError:
        print(f"Warning: Skipping folder '{folder.name}' - error decoding 'training_data.json'")
        continue
    except Exception as e:
        print(f"Warning: Skipping folder '{folder.name}' - unexpected error reading json: {e}")
        continue

    source = folder.name # Use folder name as the article ID
    article_sentence_indices = [] # Store indices *within the global sent_rows* for this article

    for sid, s in sentence_dict.items():
        # Basic validation of sentence structure
        if not all(k in s for k in ["words", "gaze_x", "gaze_y", "gaze_dur"]):
             print(f"Warning: sentence {sid} in {source} missing required keys. Skipping sentence.")
             continue

        sentence_txt = " ".join(s["words"])

        # Validate gaze data consistency and non-emptiness
        if not (len(s['gaze_x']) == len(s['gaze_y']) == len(s['gaze_dur'])):
            print(f"Warning: sentence {sid} in {source} has inconsistent gaze array lengths. Skipping sentence.")
            continue
        if len(s['gaze_x']) == 0:
             print(f"Warning: sentence {sid} in {source} has empty gaze data. Skipping sentence.")
             continue

        # Create the sentence row
        row = {
            # Normalize gaze coordinates and convert duration
            "gaze_x":   [x / screen_width for x in s["gaze_x"]],
            "gaze_y":   [y / screen_height for y in s["gaze_y"]],
            "gaze_dur": [d * metric for d in s["gaze_dur"]],
            "sentence": sentence_txt,
            "article_id": source
        }
        sent_rows.append(row)
        # Record the global index of the sentence just added
        article_sentence_indices.append(len(sent_rows) - 1)

    # Only add article if it contained valid sentences
    if article_sentence_indices:
         article_rows.append({
            "sentence_indices": article_sentence_indices, # List of indices into sent_rows
            "summary": "", # Summary is empty as per requirement
            "article_id": source
        })
    else:
        print(f"Warning: No valid sentences found in folder '{source}'. Skipping article.")


if not sent_rows:
    print("Error: No valid sentence data found in any processed folder. Exiting.")
    exit()

if not article_rows:
    print("Error: No valid article data generated (check sentence processing warnings). Exiting.")
    exit()

print(f"\nCollected {len(sent_rows)} sentences from {len(article_rows)} articles.")

# --- Dataset Creation (Test Split Only) ---

# Create Stage-1 dataset (sentences)
print(f"Creating Stage-1 dataset (sentences) at '{OUT1}'...")
Path(OUT1).mkdir(parents=True, exist_ok=True)
DatasetDict({
    "test": Dataset.from_list(sent_rows),
}).save_to_disk(str(OUT1)) # Use str() for older datasets versions if needed

# Create Stage-2/3 dataset (articles)
# The sentence_indices already point to the correct rows in the sent_rows list,
# which is exactly what the 'test' split of the stage 1 dataset contains.
print(f"Creating Stage-2/3 dataset (articles) at '{OUT2}'...")
Path(OUT2).mkdir(parents=True, exist_ok=True)
DatasetDict({
    "test": Dataset.from_list(article_rows),
}).save_to_disk(str(OUT2)) # Use str() for older datasets versions if needed

print("\n✓ Both datasets saved with only a 'test' split.")

# --- Preview Generation ---

# --- Article-level preview ---
preview_art = []
n_art = len(article_rows)
n_sent = len(sent_rows) # Total sentences collected is the sum for the 'test' split
preview_art.append(f"== TEST ({n_art} articles, {n_sent} sentences) ==\n")
for art in article_rows:
    preview_art.append(f"article_id: {art['article_id']:<20} " # Adjust padding if needed
                       f"sentences: {len(art['sentence_indices'])}\n")
preview_art.append("\n")

preview_art_path = Path(OUT2, "preview.txt")
preview_art_path.write_text("".join(preview_art))
print(f"✓ Article-level preview → {preview_art_path}")

# --- Sentence-level preview ---
preview_sent = []
preview_sent.append(f"== TEST ({len(sent_rows)} sentences) ==\n")
# Cap preview for brevity if needed, e.g., enumerate(sent_rows[:500])
for idx, r in enumerate(sent_rows):
    snippet = shorten(r["sentence"], width=70, placeholder="…")
    preview_sent.append(f"[{idx:05}] article:{r['article_id']:<20}  {snippet}\n") # Adjust padding
preview_sent.append("\n")

preview_sent_path = Path(OUT1, "preview.txt")
preview_sent_path.write_text("".join(preview_sent))
print(f"✓ Sentence-level preview → {preview_sent_path}")

print("\nProcessing complete.")