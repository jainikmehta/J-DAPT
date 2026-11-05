# ABOships-PLUS Dataset (Maritime Robotics)

## 📌 Overview
**ABOships-PLUS** is a high-resolution maritime perception dataset developed by Åbo Akademi University (Iancu et al., JMSE 2023). In the J-DAPT framework, it represents the **Target Domain ($D_T$)** for **maritime robotics and autonomous surface vessels**, capturing real onboard camera views of open waterways, harbor channels, buoys, and vessel traffic.

- **Total Images**: 9,880 high-resolution coastal & open sea images
  - **Train Split**: 6,916 images in `images/train/`
  - **Val Split**: 1,482 images in `images/val/`
  - **Test Split**: 1,482 images in `images/test/`
- **Annotations**: 11 vessel and maritime obstacle classes in `annotations/`
- **Paper**: *A Benchmark for Maritime Object Detection with CenterNet on an Improved Dataset, ABOships-PLUS* (JMSE 2023)

---

## 📁 Directory Structure

```
datasets/aboships/
├── README.md                   # This documentation file
├── annotations/                # COCO-format ground truth annotations
│   ├── train.json              # 6,916 image object bounding boxes & vessel classes
│   ├── val.json                # 1,482 image validation annotations
│   └── test.json               # 1,482 image test annotations
└── images/                     # Raw onboard coastal camera frames
    ├── train/                  # 6,916 images (201806260722_001.png ...)
    ├── val/                    # 1,482 validation images
    └── test/                   # 1,482 test images
```

---

## 🏷️ Maritime Obstacle Classes
Annotations cover 11 safety-critical categories encountered by autonomous boats:
1. `Boat`
2. `Vessel` / `Cargo`
3. `Ferry`
4. `Sailboat`
5. `Kayak`
6. `Buoy`
7. `Seamark`
8. `Motorboat`
9. `Miscellaneous Maritime Obstacle`

---

## 🎯 Role in J-DAPT
In J-DAPT:
1. **Preprocessing (`preprocessing.py`)**: Groups sequential nautical frames by timestamp prefix, compiles short video sequences (`.mp4`), and queries server VLMs for maritime goals/targets (`_jb.txt`) and nominal queries (`_bq.txt`).
2. **Domain Adaptation (`jdapt-boat.ipynb`)**: Aligns general multimodal jailbreaks (`JB28K`) with maritime perception features (`ABOships` + `LaRS`) to train `boat_classifier.pth`, preventing attacks that attempt to crash autonomous vessels into buoys or violate maritime navigation rules.

