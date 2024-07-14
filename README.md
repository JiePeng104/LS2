# LS2: Boosting Hidden Separation for Backdoor Defense with Learning Speed-driven Label Smoothing

## Overview
Implementation of Learning Speed-driven Label Smoothing (LS^2).

Our implementation is mainly developed from three public repository:

1. [BackdoorBench-V1](https://github.com/SCLBD/BackdoorBench/tree/v1)
2. [SPECTRE-Defense](https://github.com/SewoongLab/spectre-defense)
3. [Adaptive Attacks against Latent Separation Defense](https://github.com/Unispac/Circumventing-Backdoor-Defenses)

## Attacks

Datasets: CIFAR-10, CIFAR-100, GTSRB, ImageNet100

Note: [ImageNet100](https://www.kaggle.com/datasets/ambityga/imagenet100/data) is a subset of [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/2012/).


|                  | File name                                                   |
|------------------|-------------------------------------------------------------|
| BadNets          | [badnets_attack_ls.py](./attack/badnet_attack_ls.py)       |
| Blended          | [blended_attack_ls.py](./attack/blended_attack_ls.py)       |
| LF               | [lf_attack_ls.py](./attack/lf_attack_ls.py)                    |
| ISSBA            | [ssba_attack_ls.py](./attack/ssba_attack_ls.py)                    |
| Adap_Patch       | [adap_patch_ls.py](./attack/adap_patch_ls.py)                    |
| Adap_Blended     | [blended_adap_ls.py](./attack/blended_adap_ls.py)                    |
| LC               | [lc_attack_ls.py](./attack/lc_attack_ls.py)                    |
| SIG              | [sig_attack_ls.py](./attack/sig_attack_ls.py)                    |

To perform LC and ISSBA attacks, please refer to [BackdoorBench-V1](https://github.com/SCLBD/BackdoorBench/tree/v1) 
for necessary resources.

## Defenses


|                  | File name                                                   |
|------------------|-------------------------------------------------------------|
| ABL          | [abl.py](./defense/abl/abl.py)       |
| ANP         | [anp.py](./defense/anp/anp.py)        |
| AWM               | [awm.py](./defense/awm/awm.py)                    |
| I-Bau            | [i-bau.py](./defense/i-bau/i-bau.py)                   |
| NC       | [nc.py](./defense/nc/nc.py)                    |
| DBD       | [dbd.py](./defense/dbd/dbd.py)                    |
| AC, SS and SPCTRE    | [quantum.py](./defense/filter/quantum.py),   [run_filters.jl](./defense/filter/run_filters.jl)                   |

The code of latent-separation-based defense (i.e., AC, SS, SPECTRE) is implemented in Julia.
The original requirements can be found at [SPECTRE-Defense](https://github.com/SewoongLab/spectre-defense).
*Here, we recommend you use [Julia-up](https://github.com/JuliaLang/juliaup) to install Julia 1.6:*

    curl -fsSL https://install.julialang.org | sh
    juliaup add +1.6


For performing D-BR defense, 
we recommend the open-source [BackdoorBench-V2](https://github.com/SCLBD/BackdoorBench), 
which has a different architecture than [BackdoorBench-V1](https://github.com/SCLBD/BackdoorBench/tree/v1) we use here.

##Usage
The scripts can be found in the folder [/sh](sh).
We provide an example for performing Adap-Blended attack and the corresponding defenses. 
```bash
# Before running a script, please change the working directory to this project
cd ../backdoor-ls2-code
# Train a model using Cross-Entropy (poison rate = 0.003)
# The result can be found in ./record/<save_folder_name>
python ./attack/blended_adap_ls.py --yaml_path ../config/attack/blended_adap/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name blended_adap_pre_p-0.003_cover-0.003 --cover_rate 0.003
# Train a model using Standard Label Smoothing (smoothing rate = 0.1)
python ./attack/blended_adap_ls.py --yaml_path ../config/attack/blended_adap/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name blended_adap_LS_sr-0.1_p-0.003_cover-0.003 --cover_rate 0.003 --smooth_rate 0.1
# Train a model using LS2 (Warm up epoch = 10, smoothing upper bound = 0.2)
python ./attack/blended_adap_ls.py --yaml_path ../config/attack/blended_adap/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --save_folder_name blended_adap_LS2_pre_p-0.003_cover-0.003 --cover_rate 0.003 --dynamic_ls True --warmup_end 10 --p_rate 0.2 --n_rate 0.0 --cover_rate 0.003 --log_aum True --smooth_rate 0.1

# Try to remove poison samples using AC, SS and SPECTRE
# The TSNE visualization results are saved in ./record/<save_folder_name>/tsne_visual
python ./latent_separation_defense/adap_patch_filter.py --yaml_path ../config/attack/blended_adap/cifar10_p-0.003.yaml --dataset cifar10 --dataset_path ../data --load_path blended_adap_LS2_pre_p-0.003_cover-0.003 --save_folder_name blended_adap_LS2_pre_p-0.003_cover-0.003 --tsne_visual True

# Other defenses:
python ./defense/anp/anp.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/anp/cifar10.yaml --dataset cifar10
python ./defense/awm/awm.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/awm/cifar10.yaml --dataset cifar10
python ./defense/nc/nc.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/nc/cifar10.yaml --dataset cifar10
python ./defense/i-bau/i-bau.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/i-bau/cifar10.yaml --dataset cifar10
python ./defense/abl/abl.py --result_file blended_adap_LS2_pre_p-0.003_cover-0.003 --yaml_path ./config/defense/abl/cifar10.yaml --dataset cifar10
```
