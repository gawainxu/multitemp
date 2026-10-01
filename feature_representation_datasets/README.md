# Datasets from "What shapes feature representations? Exploring datasets, architectures, and training"
  
This repository contains the image datasets used in the paper ["What shapes feature representations? Exploring datasets, architectures, and training"](https://arxiv.org/abs/2006.12433) by Katherine Hermann* and Andrew Lampinen* (contributed equally).
  
# Requirements
	
python==3.7.4, pytorch==1.2.0, torchvision==0.4.0, numpy==1.17.0, matplotlib==3.1.1, scipy==1.3.1, torchsummary, pickle, pillow==6.1.0, seaborn
	
# Generating datasets
	
Our datasets are generated in two steps: 
	
1) Generate stimuli.
2) Split into multiple train/val splits (for correlated trifeature, this stage is where correlations are introduced).
	
## Navon
	
The Navon stimuli and splits as used in our experiments appear in `navon_twice_rotated/` and `dataset_specs/navon_twice_rotated/`, respectively. These are adapted from Hermann et al. 2020, based on stimuli created by Navon 1977.

To generate Navon stimuli:
	
```python
import generate_navon_stims
	
generate_navon_stims.make_stims(
  savedir="./navon_stimuli",
  twice_rotated=True,
	font_path="/Library/Fonts/Arial/Arial_Bold.ttf")  # update as needed
```
	
To generate Navon splits:
	
```python
import split_datasets
	
split_datasets.main_navon(
  save_dir="dataset_specs/navon_twice_rotated/",
  navon_stims_dir="navon_stimuli")
```
	
To split into datasets with different percentages of training data, see `sample_percent_subsets.py`.
	
## Trifeature
	
To generate Trifeature stimuli:
	
```python
import color_texture_shape_stims
	
color_texture_shape_stims.save_stimuli(
  output_directory="./color_texture_shape_stimuli/")
```
	
To generate Trifeature splits:
	
```python
import split_datasets
	
# uncorrelated
split_datasets.main_cst(save_dir="dataset_specs/cst_uncorrelated/",
	cst_stims_dir="color_texture_shape_stimuli")
	
# correlated
split_datasets.main_cst_correlated(
save_dir_root="dataset_specs/cst_correlated/",
	cst_stims_dir="color_texture_shape_stimuli")
	
```

# References

Navon, D. (1977). Forest before trees: The precedence of global features in visual perception. Cognitive psychology, 9(3), 353-383.

Hermann, K. L., Chen, T. and Kornblith, S. (2020). The origins and prevalence of texture bias in convolutional neural networks. Advances in Neural Information Processing Systems 33.
