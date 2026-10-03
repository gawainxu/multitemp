# MultiTemp

Official implementation for the project:

**What Makes Representations Good for Open-Set Recognition?**

This repository contains code for training the multi-temperature supervised contrastive learning models, extracting learned representations, and evaluating classification accuracy and open-set recognition performance.

## Requirements

The main dependencies are:

- PyTorch
- torchvision
- NumPy
- scikit-learn
- SciPy
- Matplotlib

## Training

To train a model, for example on CIFAR-10:

```bash
python main_supcon.py 
   --batch_size 256 
   --epochs 600 
   --learning_rate 0.001
   --model "resnet_multi" 
   --datasets "cifar10" 
   --method "SupCon" 
   --trail 0 
   --temp1 1.0 
   --temp2 1.0 
   --temp3 1.0
```

The `--temp1`, `--temp2`, `--temp3` arguments specify the temperatures used for the supervised multi-temperature contrastive learning.

The "--trail" argument specifies the index of the evaluation protocol.

## Feature Extraction

After training, representations can be extracted and saved using `feature_reading.py`.

For example:

```bash
python feature_reading.py \
    --datasets "cifar10" \
    --trail 0 \
    --model "resnet18" \
    --model_path PATH_TO_MODEL \
    --if_train "train" \
    --method "SupCon" \
    --feature_save SAVE_PATH \
```

The `--if_train` argument determines which subset is used for feature extraction:

- `train`: training exemplars
- `testing_known`: known/inlier test samples
- `testing_unknown`: unknown/outlier test samples

Run the script separately for the required subsets and specify the corresponding output paths with `--feature_save`.

## Accuracy and AUROC Evaluation

After extracting the exemplar, known-test, and unknown-test features, classification accuracy and open-set AUROC can be evaluated using:

```bash
python main_testing_multi.py \
    --datasets "cifar10" \
    --ensembles 1 \
    --K 3 \
    --model "resnet18" \
    --exemplar_features_path PATH_TO_EXEMPLAR_FEATURES \
    --testing_known_features_path PATH_TO_INLIER_TESTING_FEATURES \
    --testing_unknown_features_path PATH_TO_OUTLIER_TESTING_FEATURES \
    --num_classes 6 \
    --trail 0
```

The three feature paths correspond to:

- `--exemplar_features_path`: features extracted from training exemplars
- `--testing_known_features_path`: features from known/inlier test samples
- `--testing_unknown_features_path`: features from unknown/outlier test samples

The evaluation reports closed-set classification accuracy on known samples and AUROC for distinguishing known from unknown samples.

## Workflow

The basic experimental pipeline is:

```text
Train model
    ↓
Extract exemplar features
    ↓
Extract known test features
    ↓
Extract unknown test features
    ↓
Evaluate classification accuracy and OSR AUROC
```