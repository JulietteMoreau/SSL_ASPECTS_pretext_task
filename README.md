# New pretext task for neuroimaging self-supervised learning

Repository for the EMBC 2025 paper, Self-supervised learning for stroke lesion segmentation on CT: a new pretext task for 3D neuroimaging.
It contains codes for the proposed pretext task based on the segmentation of the ASPECTS zones, specifically for stroke lesion segmentation target task, and two reference pretext tasks: rotation between 4 angles and inpainting, and codes for training the target tasks based on the three pretext tasks. All of the codes are based on the 3D U-Net architecture.

# Data preprocessings and organization

Data should be separated between pretext and target tasks. Preprocessing codes are proposed for the three pretext tasks. Images can be resized to the same size, and flipped to place all the lesions on the same size if reference is known, with `resize_flip_data.py`. The script `random_rotate.py` creates the dataset for rotation pretext taks, the script `create_white_patch.py` created the dataset for inpainting pretext task.
The ASPECTS segmentation pretext task is based on registration of the ASPECTS zones segmented on an ATLAS brain (see `data/ATALS`) on all volumes of the pretext dataset. This is performed with ANTs and the script `run_ants_atlas.sh`, and postprocess of the ASPECTS mask can be done with `postprocess_ASPECT.py`.

For the trainings, data is supposed to be organized as follows:

```
data/
├── pretext/
│   ├── image/
│   │   ├── train/
│   │   │   └── img1.nii.gz
│   │   ├── validation/
│   │       └── img2.nii.gz
│   └── reference/
│       ├── train/
│       │   └── img1.nii.gz
│       ├── validation/
│           └── img2.nii.gz
├── target/
│   ├── image/
│   │   ├── train/
│   │   │   └── img1.nii.gz
│   │   ├── validation/
│   │   │   └── img2.nii.gz
│   │   └── test/
│   │       └── img3.nii.gz
│   └── reference/
│       ├── train/
│       │   └── img1.nii.gz
│       ├── validation/
│       │   └── img2.nii.gz
│       └── test/
│           └── img3.nii.gz
```


# Pretext tasks

Pretext tasks are trained with `pretext_task/run_SSL_pretext.pbs`, and evaluated with the scrpts `pretext_task/evaluation_XXX.py`.

# Target tasks

Target tasks are trained after the pretext tasks, fine-tuning the weights, with `target_task/run_SSL_cible.pbs` and evaluation with `evaluation_checkpoint.py`.
