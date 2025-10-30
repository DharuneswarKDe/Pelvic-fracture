
## Structure-Focused Contrastive Learning for Pelvic Fracture Detection

## Overview
Pelvic fractures are complex injuries that can be hard to spot because of the pelvis's structural complexity and irregular morphology. Manual assessment by radiologists is effective but time-consuming and variable. This project offers a structure-aware AI pipeline that uses segmentation, symmetry alignment, and contrastive analysis to detect possible pelvic fractures from CT scans.

Key ideas:
- Segment the pelvis into three anatomical fragments (sacrum, left innominate, right innominate) using an Encoder–Decoder (U-Net) model.
- Mirror and align pelvic halves using Iterative Closest Point (ICP).
- Compare voxel-level density (Hounsfield units) across aligned halves to flag potential fracture areas.
- Use contrastive learning to improve the model's robustness on small or imbalanced datasets.

This repository contains a single Google Colab notebook that runs the full pipeline end-to-end for easy reproduction.

---

## Dataset

Primary dataset used:

* **FracSegNet** by Y. Liu et al. — GitHub: [https://github.com/YzzLiu/FracSegNet](https://github.com/YzzLiu/FracSegNet)

We also used a custom clinical dataset of 100+ preoperative CT scans for validation and additional testing.

**Important:** Please follow the original dataset license and cite the FracSegNet authors if you reuse their data or code.

---

**Dataset (FracSegNet):**

```
Y. Liu et al., "FracSegNet: A Deep Learning Framework for Fracture Segmentation and Classification in Pelvic CT Images", 2021.
GitHub: https://github.com/YzzLiu/FracSegNet
```

Include FracSegNet authors when sharing results or derivative works.

---

## Results

* The pipeline produced promising fracture localization results on combined datasets.
* Reported performance (approximate): ~90% detection accuracy on validation scans (see notebook evaluation cells for details).
* Visual outputs (in `results/`) include segmentation masks, aligned comparisons, and heatmaps marking suspect areas.

---


