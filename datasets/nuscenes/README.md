# nuScenes Dataset (Autonomous Driving)

## 📌 Overview
**nuScenes** is an industry-standard multimodal autonomous driving dataset developed by Motional (Caesar et al., CVPR 2020). It provides full 360-degree sensor coverage (6 cameras, 1 LiDAR, 5 radars, GPS/IMU, 3D bounding boxes) collected in Boston and Singapore. In the J-DAPT framework, nuScenes camera sequences are used to ground **RoboPAIR multi-turn robotic jailbreak attacks** against Vision-Language Models.

- **Version**: `v1.0-trainval` (Blobs 01 + Full Metadata)
- **Front Camera Images**: 3,376 keyframe images in `samples/CAM_FRONT/`
- **Total Metadata Records**: 13 relational tables in `v1.0-trainval/`
- **Paper**: *nuScenes: A Multimodal Dataset for Autonomous Driving* (CVPR 2020)

---

## 📁 Directory Structure

```
datasets/nuscenes/
├── README.md                   # This documentation file
├── LICENSE                     # nuScenes terms of use and non-commercial license
├── v1.0-trainval/              # Relational JSON metadata tables
│   ├── scene.json              # Drive scene logs and descriptions
│   ├── sample.json             # Keyframe timestamps and annotations
│   ├── sample_data.json        # Pointers to camera, radar, and LiDAR sensor blobs
│   ├── sample_annotation.json  # 3D bounding boxes (pedestrians, vehicles, barriers)
│   ├── ego_pose.json           # Vehicle coordinates, speed, and heading
│   └── calibrated_sensor.json  # Camera intrinsics and extrinsics
├── samples/                    # Synchronized sensor keyframes (2 Hz)
│   ├── CAM_FRONT/              # Frontal camera views (3,376 images)
│   ├── CAM_FRONT_LEFT/         # Front-left camera views
│   ├── CAM_FRONT_RIGHT/        # Front-right camera views
│   ├── CAM_BACK/               # Rear camera views
│   ├── CAM_BACK_LEFT/          # Rear-left camera views
│   └── CAM_BACK_RIGHT/         # Rear-right camera views
├── sweeps/                     # Intermediate sensor frames between keyframes
└── maps/                       # High-definition vector map representations
```

---

## 🎯 Role in J-DAPT
In J-DAPT:
1. **Preprocessing (`preprocessing.py`)**: Uses the nuScenes devkit (`nuscenes.nuscenes.NuScenes`) to render 20-frame `.mp4` driving videos from camera feeds and generate goal/target prompts (`jb.txt` and `bq.txt`).
2. **RoboPAIR Jailbreaks (`jailbreak.sh`)**: Red-teaming attacker LLMs iteratively probe the driving system with adversarial prompts on nuScenes video scenes.
3. **Domain Alignment (`jdapt-car.ipynb`)**: nuScenes visual embeddings are combined with Wayve to form the target autonomous vehicle covariance matrix for CORAL adaptation.

