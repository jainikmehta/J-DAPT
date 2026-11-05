# Wayve / LingoQA Dataset (Autonomous Driving)

## 📌 Overview
The **Wayve LingoQA** dataset is a large-scale visual question-answering benchmark designed for autonomous driving reasoning (Marcu et al., ECCV 2024, Wayve). In the J-DAPT framework, it represents the **Target Domain ($D_T$)** for autonomous vehicles, supplying camera perspectives from real-world driving sequences and natural driving instructions.

- **Total Annotations**: 148,506 driving QA rows in `train.parquet`
- **Unique Driving Inquiries**: 63,673 distinct driver questions
- **Total Drive Sequences**: 3,508 driving video segments
- **Extracted Images**: 17,540 frontal road scene images (5 frames per sequence)
- **Paper**: *LingoQA: Visual Question Answering for Autonomous Driving* (ECCV 2024)

---

## 📁 Directory Structure

```
datasets/wayve/
├── README.md                   # This documentation file
├── train.parquet               # Master parquet containing driver QA, video IDs & bounding data
├── images.zip                  # Original compressed archive of frontal road images
└── images/
    └── train/                  # Extracted sequence directories
        ├── 00001a600abbc608.../ # Sequence folder (5 consecutive temporal frames)
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   ├── 2.jpg
        │   ├── 3.jpg
        │   └── 4.jpg
        └── ... (3,508 sequence folders, 17,540 images)
```

---

## 🏷️ Metadata Schema (`train.parquet`)

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `segment_id` | string | Unique 32-character drive sequence identifier | `403fcde1af301b3e5d271176db3ea81c` |
| `question` | string | Driver situational inquiry | *"Is there any obstacle in your way?"*, *"What action are you taking?"* |
| `answer` | string | Ground truth driver reasoning | *"I am slowing down because a pedestrian is crossing."* |
| `images` | list[string] | Relative paths to temporal camera frames | `['images/train/403f.../0.jpg', ...]` |

---

## 🎯 Role in J-DAPT
In J-DAPT, `wayve` provides realistic camera views and benign autonomous driving context:
1. **Preprocessing (`preprocessing.py`)**: Uses server VLMs to convert nominal road queries into red-team driving attacks (`train_redteam.parquet`).
2. **Offline Adaptation (`train_car_classifier.py`)**: Provides real vehicle camera inputs to ground the 8-head `CrossAttentionFusion` model on road assets (crosswalks, pedestrians, lane markings, lead vehicles).

