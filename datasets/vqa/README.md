# DAQUAR Dataset (Visual Question Answering)

## 📌 Overview
**DAQUAR** (*DAtaset for QUestion Answering on Real-world images*) is an indoor scene visual question-answering benchmark introduced by Mateusz Malinowski and Mario Fritz (NeurIPS 2014, MPI Informatics). It serves as the primary **Source Domain ($D_S$)** for **benign / nominal queries** in the J-DAPT framework, providing a baseline of legitimate, standard visual reasoning.

- **Total QA Pairs**: 12,468 question-answer pairs
  - **Train Split**: 6,794 raw pairs (expanded to 9,974 with augmentations)
  - **Test Split**: 5,674 raw pairs (expanded to 2,494 evaluation sets)
- **Total Images**: 1,449 indoor RGB images (from the NYU-Depth V2 dataset)
- **Average Queries/Image**: ~8.6 questions per scene
- **Paper**: *A Multi-World Approach to Question Answering About Real-World Scenes Based on Uncertain Input* (NeurIPS 2014)

---

## 📁 Directory Structure

```
datasets/vqa/
├── README.md                   # This documentation file
├── data.csv                    # Master table containing all 12,468 QA pairs & image IDs
├── data_train.csv              # Training split QA pairs
├── data_eval.csv               # Evaluation / test split QA pairs
├── all_qa_pairs.txt            # Raw text dump of all question-answer pairs
├── answer_space.txt            # Vocabulary of valid answer labels
├── train_images_list.txt       # List of 795 training image IDs
├── test_images_list.txt        # List of 654 test image IDs
├── good_examples.txt           # Verified reference examples from the original benchmark
└── images/                     # 1,449 indoor RGB scene images (image1.png ... image1449.png)
```

---

## 🏷️ Metadata Schema (`data.csv`)

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `question` | string | Natural language question about objects in the scene | *"what is on the left side of the white oven on the floor?"* |
| `answer` | string | Ground truth object or scene attribute | `garbage_bin`, `table`, `chair` |
| `image_id` | string | Identifier referencing image filename in `images/` | `image1` $\rightarrow$ `images/image1.png` |

---

## 🔍 Why Multiple Questions Per Image?
Unlike single-label classification datasets, VQA datasets evaluate situational awareness by querying multiple objects, spatial relationships, and colors within the same environment. In DAQUAR, each of the 1,449 images has between 1 and 31 associated questions (e.g. `image8.png` has 31 distinct questions regarding different household items).

---

## 🎯 Role in J-DAPT
In J-DAPT, `DAQUAR` represents nominal (benign) human interactions in the source domain. During domain adaptation, features extracted from DAQUAR and JailBreakV are aligned against robotic target datasets (`Wayve`, `nuScenes`, `ABOships`, `LaRS`) to train the defense classifiers.

