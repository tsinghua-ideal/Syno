
from torchvision.datasets import HMDB51
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm
from torchvision.transforms import transforms, v2

train_data = HMDB51(
    root=os.path.join("/cephfs/suzhengyuan/data", "hmdb/videos"), 
    annotation_path=os.path.join("/cephfs/suzhengyuan/data", "hmdb/annotations"), 
    frames_per_clip=16, 
    step_between_clips=16, 
    num_workers=16,
    transform= transforms.Compose([
        # transforms.Resize((112, 112)),
        transforms.ConvertImageDtype(torch.float),
    ]),
    output_format="THWC", 
    train=True
)

sum_x = 0
sum_x2 = 0
n = 0

# train_dataloader = DataLoader(
#     train_data,
#     batch_size=16,
#     shuffle=False,
#     num_workers=8,
#     pin_memory=True,
#     persistent_workers=False,
# )

for video, _, _ in tqdm(train_data):
    n += video.numel() / video.size(-1)

for video, _, _ in tqdm(train_data):
    v = video.view(-1, 3)
    sum_x += v.sum(0) / n
    sum_x2 += (v * v).sum(0) / n

mean = sum_x
stdev = (sum_x2 - mean * mean).sqrt()
print(f"mean={mean}, stdev={stdev}")