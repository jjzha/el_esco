# ESCO Entity Linking

This repository is for the paper 

**Entity Linking in the Job Market Domain**

Mike Zhang, Rob van der Goot, and Barbara Plank. To appear at EACL Findings 2024.

---

We make use of two separate models, BLINK and GENRE. Each have their separate folder. We suggest you to read each README of the respective repository to use the models and how to preprocess the data.

In each folder, we left an `environment.yml` to reproduce our experiments. We suggest to create a `conda` environment. Unfortunately, you have to use the environments separatly.

To install:

```
# BLINK
conda create env -f BLINK/environment.yml

# GENRE
conda create env -f GENRE/environment.yml
```

Once you followed the installation instructions in both BLINK and GENRE. You can train models.

# Data
In this repository, you can find a `.tar` (`data_entity_linking_esco.tar`) file with the respective data for each model. We suggest putting the data in a `/data/` folder per model directory. 

# Training
In both BLINK and GENRE, we left a folder `ESCO_scripts` for the ESCO-specific experiments.

To run these scripts. You can simply call them:

```
# BLINK
cd BLINK
bash ESCO_scripts/train_blink_biencoder.sh
bash ESCO_scripts/train_blink_biencoder_pretrained.sh

# GENRE
cd GENRE
bash ESCO_scripts/train_bart.sh
bash ESCO_scripts/train_genre_pretrained.sh
```

The `*_pretrained.sh` files are for further fine-tuning the models released by both BLINK and GENRE.

# Evaluation
Also in the `ESCO_scripts` folders of both models, you can find the evaluation scripts `BLINK/ESCO_scripts/eval_blink_biencoder.sh` and `GENRE/ESCO_scripts/evaluate_genre.sh`. 

Once you trained your models. These should be able to run. By default, it should also create the predictions.
