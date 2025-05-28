import pandas as pd

data_dir = 'results_real_stage3_separated'
# data_dir = 'results_stage3_separated'

df = pd.read_csv(f"{data_dir}/generated_summaries_comparison.csv")

def extract_summary(summary: str):
    # start = summary.find('\n')
    start = 0
    end = summary.find('Human:')
    if end == -1:
        end = summary.find('.Can you')
    if end != -1:
        end += 1
    else:
        end = len(summary)
    print(summary[start:end].strip())
    return

for _, row in df.iterrows():
    # Extract the values from the row
    # article_id,article,summary_gaze,summary_base
    summary_gaze = row['summary_gaze']
    summary_base = row['summary_base']

    print(f"Generated Summary: {summary_gaze}")
    print()
    # print(f"Reference Summary: {summary_base}")
    print('reference summary')
    extract_summary(summary_base)
    print()
    print()