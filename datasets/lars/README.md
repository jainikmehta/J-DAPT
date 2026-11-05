# LaRS Dataset (Maritime Panoptic Obstacle Detection)

## 📌 Overview
**LaRS** (*Lake, River, and Sea*) is a large-scale panoptic maritime obstacle detection and semantic segmentation dataset (Žust et al., ICCV 2023, ViCoS Lab). In the J-DAPT framework, it represents the **continuous temporal video stream domain ($D_T$)** for autonomous boats and maritime robots across diverse water bodies (lakes, rivers, coastal regions).

- **Coverage**: Lakes, rivers, canals, open sea, under variable weather, glare, and water conditions
- **Subsets**: Sequential frames in `train/`, `val/`, and `test/` splits
- **Paper**: *LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark* (ICCV 2023)

---

## 📁 Directory Structure

```
datasets/lars/
├── README.md                   # This documentation file
├── train/
│   └── images_seq/             # Consecutive video sequences for training
├── val/
│   └── images_seq/             # Validation sequence frames
└── test/
    └── images_seq/             # Test sequence frames
```

---

## 🏷️ Sequence Naming Convention
Image filenames inside `images_seq/` follow a structured prefix-indexed format:
```
<water_body>_<scene_id>_<timestamp>_<frame_id>.jpg
```
For example: `seq_01_00000.jpg`, `seq_01_00001.jpg` ...

---

## 🎯 Role in J-DAPT
In J-DAPT:
1. **Video Sequence Preprocessing (`preprocessing.py`)**: Gathers ordered image frames from `images_seq`, compiles MP4 nautical clips via `imgs2mp4()`, and prompts the VLM to produce maritime goals and targets (`_jb.txt` and `_bq.txt`).
2. **Maritime Jailbreak Detection (`jdapt-boat.ipynb`)**: Combined with `ABOships-PLUS` to form the target maritime domain representation, training the adapted boat defense classifier to prevent collision-course overrides in maritime autonomy.

