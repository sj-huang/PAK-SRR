# PAK-SRR
Super-Resolution Reconstruction of Fetal Brain MRI with Prior Anatomical Knowledge


# Step 0: Download Pre-trained Models

./1_samonaifbs/models/checkpoint_dynUnet_DiceXent.pt https://zenodo.org/record/4282679#.X7fyttvgqL5

./1_samonaifbs/models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth


# Step 1: Brain Extraction

run python ./1_samonaifbs/src/inference/step1_main.py

# Step 2: Images Reorientation

run python ./2_reorientation.py

# Step 3: Tissue Segmentation and Distance map Calculation

run python ./3_tissue_seg/step3_main.py

# Step 4: Super-Resolution Reconstruction 1 (Baseline Method: NiftyMIC)

run python ./4_niftymic/baseline_main.py

# Step 4: Super-Resolution Reconstruction 2 (Our Method: PAK-SRR)

run python ./4_paksrr/PAK_SRR_main.py
