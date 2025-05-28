# stage1_data.py
from datasets import load_from_disk
from torch.utils.data import Dataset
import torch
from config import EyeT5Config

class GazeSentenceDS(Dataset):
    """
    One sentence → (numeric_tensor  [L,3],  sentence_str,  L)
    """
    def __init__(self, hf_dir: str, split: str, cfg: EyeT5Config):
        self.ds  = load_from_disk(hf_dir)[split]
        self.cfg = cfg

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        x_num = torch.tensor(list(zip(row["gaze_x"],
                                      row["gaze_y"],
                                      row["gaze_dur"])),
                             dtype=torch.float)
        return x_num, row["sentence"], len(x_num)


def collate(batch):
    nums, sents, lens = zip(*batch)
    max_L = max(lens)
    F_num = nums[0].shape[-1]
    pad   = torch.zeros(len(nums), max_L, F_num)
    mask  = torch.zeros(len(nums), max_L, dtype=torch.bool)
    # print(pad.shape, mask.shape)
    # print(nums, sents, lens)
    for i,(x,l) in enumerate(zip(nums,lens)):
        pad[i,:l]  = x
        mask[i,:l] = True
    return pad, mask, list(sents), torch.tensor(lens)