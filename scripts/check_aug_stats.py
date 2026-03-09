from pathlib import Path
from collections import defaultdict

OUT = Path(r"C:\dev\active\yolo26-industrial-vision\datasets\gc10-det-aug")
NAMES = ['crease','crescent_gap','inclusion','oil_spot','punching_hole',
         'rolled_pit','silk_spot','waist_folding','water_spot','welding_line']
BEFORE = {0:56,1:212,2:292,3:445,4:265,5:76,6:695,7:122,8:283,9:423}

counts = defaultdict(int)
for lbl in (OUT / "labels" / "train").glob("*.txt"):
    for line in open(lbl):
        p = line.strip().split()
        if len(p) >= 5:
            c = int(float(p[0]))
            counts[c] += 1

train_imgs = len(list((OUT / "images" / "train").glob("*")))
val_imgs = len(list((OUT / "images" / "val").glob("*")))
print(f"Total train images: {train_imgs}")
print(f"Total val images: {val_imgs}")
print()
for i, name in enumerate(NAMES):
    b = BEFORE.get(i, 0)
    a = counts.get(i, 0)
    r = a / b if b > 0 else 0
    bar = "#" * min(50, a // 20)
    print(f"  {name:<16} {b:>4} -> {a:>5} ({r:.1f}x) {bar}")
