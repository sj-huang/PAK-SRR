# PAK-SRR
Super-Resolution Reconstruction of Fetal Brain MRI with Prior Anatomical Knowledge

Step 1: Brain extraction
run python ./1_samonaifbs/src/inference/step1_main.py

Step 2: Images Reorientation
run python ./2_reorientation.py

Step 3: Tissue segmentation and Distance map calculation
run python ./3_tissue_seg/step3_main.py

Step 4: Super-Resolution Reconstruction 1 (Baseline method: NiftyMIC)
run python ./4_niftymic/baseline_main.py

Step 4: Super-Resolution Reconstruction 2 (Our method: PAK-SRR)
run python ./4_paksrr/PAK_SRR_main.py
