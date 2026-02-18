import os
import csv
import random

TB_DIR = "data/images/TB"
NORMAL_DIR = "data/images/Normal"
CSV_PATH = "data/clinical.csv"

rows = [["image_name", "age", "fever", "cough", "weight_loss", "label"]]


for img in sorted(os.listdir(TB_DIR)):
    if img.lower().endswith(".png"):
        rows.append([
            img,
            random.randint(20, 70),
            1, 1, 1,
            1
        ])

for img in sorted(os.listdir(NORMAL_DIR)):
    if img.lower().endswith(".png"):
        rows.append([
            img,
            random.randint(20, 70),
            0, 0, 0,
            0
        ])

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f" clinical.csv created with {len(rows)-1} entries")