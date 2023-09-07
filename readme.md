# CenTime: Event-Conditional Modelling of Censoring in Survival Analysis

<p align="center">
    <a href="https://arxiv.org/abs/1234.56789">
        <img alt="arxiv1" src="https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg">
    </a>
    <a href="https://arxiv.org/abs/2203.11391">
        <img alt="arxiv2" src="https://img.shields.io/badge/arXiv-2203.11391-b31b1b.svg">
    </a>
     <a href="https://pytorch.org/get-started/previous-versions/#v191">
        <img alt="maintenance" src="https://img.shields.io/badge/PyTorch-1.9.1-ee4c2c.svg">
    </a>
     <a href="https://github.com/psf/black">
        <img alt="maintenance" src="https://img.shields.io/badge/Code%20style-black-000000.svg">
    </a>
    <a href="https://www.osicild.org/dr-about.html">
        <img alt="OSIC" src="https://img.shields.io/badge/Dataset-OSIC%20-27aae1.svg">
    </a>
</p>

## Overview

This repository contains the code for the paper
> A. H. Shahin, A. Zhao, A. C. Whitehead, D. C. Alexander, J. Jacob, D. Barber. _CenTime: Event-Conditional Modelling of Censoring in Survival Analysis_. [[arXiv]](https://arxiv.org/abs/1234.56789)

We also include the code for our prequel paper published in the International Conference on Medical Imaging with Deep Learning (MIDL) 2022
> A. H. Shahin, J. Jacob, D. C. Alexander, D. Barber. _Survival Analysis for Idiopathic Pulmonary Fibrosis using CT Images and Incomplete Clinical Data_. [[arXiv]](https://arxiv.org/abs/2203.11391) [[OpenReview]](https://openreview.net/forum?id=1234.56789) [[PMLR]](https://proceedings.mlr.press/v172/shahin22a.html)

If you use these tools or methods in your publications, please consider citing the accompanying papers with a BibTeX entry similar to the following:

```
@article{shahin2023centime,
    title={CenTime: Event-Conditional Modelling of Censoring in Survival Analysis},
    author={Shahin, Ahmed H. and Zhao, An and Whitehead, Alexander C. and Alexander, Daniel C. and Jacob, Joseph and Barber, David},
    journal={arXiv preprint arXiv:1234.56789},
    year={2023}
}

@InProceedings{shahin22a,
  title = {Survival Analysis for Idiopathic Pulmonary Fibrosis using CT Images and Incomplete Clinical Data},
  author = {Shahin, Ahmed H. and Jacob, Joseph and Alexander, Daniel C. and Barber, David},
  booktitle = {Proceedings of The 5th International Conference on Medical Imaging with Deep Learning},
  pages = {1057--1074},
  year = {2022},
  editor = {Konukoglu, Ender and Menze, Bjoern and Venkataraman, Archana and Baumgartner, Christian and Dou, Qi and Albarqouni, Shadi},
  volume = {172},
  series = {Proceedings of Machine Learning Research},
  month = {06--08 Jul},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v172/shahin22a.html},
}
```

### Requirements

We used Python 3.9.5 and PyTorch 1.9.1 in our experiments. For the rest of the requirements, please refer to the `requirements.txt` file. You can install them using the following command:
``` pip install -r requirements.txt ```.

### Data

The data used in this work is the [OSIC Dataset](https://www.osicild.org/dr-about.html). If you are an OSIC member, please register to access the repository. If not, please visit the [membership page](https://www.osicild.org/membership.html) to learn more.

In addition, the methods are quite general and can be applied to any survival analysis dataset. We are eager to see how our methods perform on other datasets and we encourage you to try them out and/or reach out with any questions or insights.

### Usage

We assume that the code is exceuted from the root directory of the repository.

#### Dataset Class

First, you will need to write a custom [pytorch dataset class](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class) that loads your data. You should place it in the `dataset.py` file. The dataset class should return a dictionary with the following keys:

- `img`: a tensor of shape `(C, H, W)` where `C` is the number of channels, `H` is the height, and `W` is the width.
- `time_to_event`: a tensor of shape `(1,)` that contains the time to event for the current sample. This is the time at which the event occurred (for an uncensored sample) or the time at which the sample was censored (for a censored sample).
- `event`: a tensor of shape `(1,)` that contains the event type for the current sample. This is `0` for a censored sample and `1` for an uncensored sample.
- `pid_sids`: a tuple of shape `(1,)` that contains the patient ID and the study ID for the current sample. This is used to group the scans of the same patient together.
- `clinical_data` (optional): a tensor of shape `(D,)` that contains the clinical data for the current sample, `D` is the number of clinical features after pre-processing. This is used to condition the model on the clinical data. If you do not have clinical data, you can ignore this key.

Preprocessing details is provided in the paper. We used the following preprocessing steps:

- CT scans
  - Resampling to a common voxel size of 1mm x 1mm x 1mm
  - Cropping to the lung area
  - Pad to the largest dimension
  - Resize to 256 x 256 x 256
  - Apply Hounsfield Unit (HU) windowing to [-1024, 150]
  - Z-score normalization
- Clinical data
  - Missing data imputation using our latent variable model (see the MIDL paper for details)
  - Z-score normalization for continuous features
  - One-hot encoding for categorical features

Changing the keys is possible but you will need to apply minimal modifications to the code, accordingly (e.g., `custom_transforms.py` and the training scripts.).

#### Missing Data Imputation

If you have missing clinical data, you will need to impute them before training. We used our latent variable model for missing data imputation. Please refer to the MIDL paper for details. The code for the model is provided in `imputation/missing_data_imputation.py`. You can use the following command to train the model:

```
python imputation/missing_data_imputation.py -data_path <path to the data> -save_path <path to save the model> -n_iter <number of iterations> -n_latent_states <number of latent states> -n_bins <number of bins to use for discretization> -tol <tolerance> -val_size <validation size percentage> -cont_feats_idx <indices of continuous features>
```

This will save the model parameters `p(h) & p(x|h)` in the specified path. The model parameters will be used for missing data imputation during training and validation (e.g., [here](https://github.com/ahmedhshahin/centime/blob/main/custom_transforms.py#L74)).

#### Training

To train the model using our CenTime method, you can use the following command:

```
python train_dist.py -loss centime ...
```

and set the rest of the arguments as needed (see `utils.py` for the full list of arguments). To use the classical censoring model, you can use the following command:

```
python train_dist.py -loss classical ...
```

To train using Cox, CoxMB, and DeepHit, you can use the following commands:

```
python train_cox.py ... # for Cox
python train_coxmb.py -K <the proportion of training data to be stored in the memory bank> ... # for CoxMB
python train_deephit.py -ranking <1 to use the ranking term, 0 otherwise> ... # for DeepHit
```

Scripts use the C-Index, MAE, and RAE metrics for evaluation (see `eval_utils.py` for details).

### Acknowledgements

This work was supported by the [Open Source Imaging Consortium](https://www.osicild.org/).
