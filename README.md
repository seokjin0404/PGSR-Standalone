# PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction
Danpeng Chen, Hai Li, [Weicai Ye](https://ywcmaike.github.io/), Yifan Wang, Weijian Xie, Shangjin Zhai, Nan Wang, Haomin Liu, Hujun Bao, [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/)
### [Project Page](https://zju3dv.github.io/pgsr/) | [arXiv](https://arxiv.org/abs/2406.06521)
![Teaser image](assets/teaser.jpg)

We present a Planar-based Gaussian Splatting Reconstruction representation for efficient and high-fidelity surface reconstruction from multi-view RGB images without any geometric prior (depth or normal from pre-trained model).  

This standalone distribution is based on the original [zju3dv/PGSR](https://github.com/zju3dv/PGSR) project. The original copyright, license, and citation requirements are retained.

## Reproducible synthetic and real-world benchmark

This repository contains the complete PGSR training, rendering, metric, and
export pipeline used for the handoff. It does not import code or environments
from any other local research repository.

### Run

Install PGSR once:

```shell
bash setup_pgsr.sh
```

For each script below, change only the clearly marked `DATA_ROOT` line at its
top. Do not change scene lists, output paths, Python files, or hyperparameters.

```shell
bash scripts/run_pgsr_synthetic.sh
```

This runs exactly `chair`, `drum`, `ficus`, `lego`, `mic`, and `ship`.

```shell
bash scripts/run_pgsr_real.sh
```

This runs exactly `fruits`, `lego`, `stair`, and `statues`. Scenes run
sequentially. Each scene completes training, checkpoint creation, rendering,
evaluation, export, and verification before the next scene starts. Set
`CUDA_VISIBLE_DEVICES` before the command to select a GPU without editing code.

Synthetic scenes need `transforms_train.json` plus validation transforms.
Real scenes need `split.json`, `colmap_workspace/run_summary.json`, and
`scale_estimation/scale_result.json`. Adapters select pattern-1 RGB without
decoding or gamma conversion.

### Results

```text
outputs/pgsr/
├── synthetic/<scene>/
│   ├── checkpoints/final.pth
│   ├── checkpoints/final_point_cloud.ply
│   ├── metrics/per_view_metrics.csv
│   ├── metrics/summary_metrics.json
│   ├── renders/val/000/
│   │   ├── gt_rgb.png
│   │   ├── pred_rgb.png
│   │   ├── rgb_error.png
│   │   ├── gt_depth.npy
│   │   ├── pred_depth.npy
│   │   ├── gt_depth_vis.png
│   │   ├── pred_depth_vis.png
│   │   ├── depth_error.png
│   │   ├── gt_normal.npy
│   │   ├── pred_normal.npy
│   │   ├── gt_normal.png
│   │   ├── pred_normal.png
│   │   └── normal_error.png
│   ├── config.json
│   └── run.log
└── real/<scene>/...
```

Every validation view is exported. Raw depth is in millimetres. Raw normals are
signed world-space vectors. PGSR camera-space normals are rotated into world
space before comparison; angular error normalizes per pixel and uses
`acos(dot)`, never `abs(dot)`.

`per_view_metrics.csv` uses this fixed schema:

```text
scene,view_id,rgb_psnr,rgb_ssim,rgb_lpips,sl_psnr,sl_ssim,depth_mae,depth_rmse,depth_median_ae,depth_p90_ae,depth_p95_ae,depth_acc_1mm,depth_acc_2mm,depth_acc_5mm,depth_abs_rel,depth_delta_1.25,depth_delta_1.25^2,depth_delta_1.25^3,normal_mean_angle,normal_median_angle,normal_p90_angle,normal_acc_10,normal_acc_20,normal_acc_22.5,normal_acc_30
```

`rgb_psnr` is peak signal-to-noise ratio in dB, `rgb_ssim` is structural
similarity, and `rgb_lpips` is perceptual distance (lower is better). Depth
`mae`, `rmse`, median AE, P90 AE, and P95 AE are millimetres. `depth_acc_*mm`
is the fraction at or below that absolute-error threshold. `depth_abs_rel` is
mean absolute error divided by GT depth. The three `depth_delta_*` fields are
fractions whose max(pred/GT, GT/pred) is below 1.25, 1.25 squared, or 1.25
cubed. Normal mean, median, and P90 are angular errors in degrees;
`normal_acc_*` is the fraction at or below the named degree threshold. All
accuracy fields are in `[0,1]`.

PGSR is RGB-only, so SL fields are `NaN`. Missing GT is also `NaN`.
Depth validity intersects finite positive GT below 10,000 mm with alpha and an
optional mask. Normal validity additionally erodes one boundary pixel and
requires a finite nonzero GT normal. Scene visualizations share a percentile
depth range, 0–10 mm error range, and 0–90 degree normal error range.

`summary_metrics.json` is the arithmetic mean across views for each metric,
independently ignoring `NaN`. Send Meng Wei the complete `outputs/pgsr/`
directory after both runs.

## Updates
- [2024.07.18]: We fine-tuned the hyperparameters based on the original paper. The Chamfer Distance on the DTU dataset decreased to 0.47.

The Chamfer Distance↓ on the DTU dataset
|     | 24| 37| 40| 55| 63| 65| 69| 83| 97|105|106|110|114|118|122|Mean|Time|
|-------|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|PGSR(Paper)|0.34|0.58|0.29|0.29|0.78|0.58|0.54|1.01|0.73|0.51|0.49|0.69|0.31|0.37|0.38|0.53|0.6h|
|PGSR(Code_V1.0)|0.33|0.51|0.29|0.28|0.75|0.53|0.46|0.92|0.62|0.48|0.45|0.55|0.29|0.33|0.31|0.47|0.5h|
|PGSR(Remove ICP)|0.36|0.57|0.38|0.33|0.78|0.58|0.50|1.08|0.63|0.59|0.46|0.54|0.30|0.38|0.34|0.52|0.5h|

The F1 Score↑ on the TnT dataset
||PGSR(Paper)|PGSR(Code_V1.0)
|-|-|-|
|Barn|0.66|0.65
|Caterpillar|0.41|0.44
|Courthouse|0.21|0.20
|Ignatius|0.80|0.81
|Meetingroom|0.29|0.32
|Truck|0.60|0.66
|Mean|0.50|0.51
|Time|1.2h|45m

## Installation

The repository contains submodules, thus please check it out with 
```shell
# Standalone repository
git clone https://github.com/seokjin0404/PGSR-Standalone.git
cd PGSR-Standalone

conda create -n pgsr python=3.8
conda activate pgsr

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #replace your cuda version
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

## Dataset Preprocess
Please download the preprocessed DTU dataset from [2DGS](https://surfsplatting.github.io/), the Tanks and Temples dataset from [official webiste](https://www.tanksandtemples.org/download/), the Mip-NeRF 360 dataset from the [official webiste](https://jonbarron.info/mipnerf360/). You need to download the ground truth point clouds from the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36). For the Tanks and Temples dataset, you need to download the reconstruction, alignment and cropfiles from the [official webiste](https://jonbarron.info/mipnerf360/). 

The data folder should like this:
```shell
data
├── dtu_dataset
│   ├── dtu
│   │   ├── scan24
│   │   │   ├── images
│   │   │   ├── mask
│   │   │   ├── sparse
│   │   │   ├── cameras_sphere.npz
│   │   │   └── cameras.npz
│   │   └── ...
│   ├── dtu_eval
│   │   ├── Points
│   │   │   └── stl
│   │   └── ObsMask
├── tnt_dataset
│   ├── tnt
│   │   ├── Ignatius
│   │   │   ├── images_raw
│   │   │   ├── Ignatius_COLMAP_SfM.log
│   │   │   ├── Ignatius_trans.txt
│   │   │   ├── Ignatius.json
│   │   │   ├── Ignatius_mapping_reference.txt
│   │   │   └── Ignatius.ply
│   │   └── ...
└── MipNeRF360
    ├── bicycle
    └── ...
```

Then run the scripts to preprocess Tanks and Temples dataset:
```shell
# Install COLMAP
Refer to https://colmap.github.io/install.html

# Tanks and Temples dataset
python scripts/preprocess/convert_tnt.py --tnt_path your_tnt_path
```

## Training and Evaluation
```shell
# Fill in the relevant parameters in the script, then run it.

# DTU dataset
python scripts/run_dtu.py

# Tanks and Temples dataset
python scripts/run_tnt.py

# Mip360 dataset
python scripts/run_mip360.py
```

## Custom Dataset
The data folder should like this:
```shell
data
├── data_name1
│   └── input
│       ├── *.jpg/*.png
│       └── ...
├── data_name2
└── ...
```
Then run the following script to preprocess the dataset and to train and test:
```shell
# Preprocess dataset
python scripts/preprocess/convert.py --data_path your_data_path
```

#### Some Suggestions:
- Adjust the threshold for selecting the nearest frame in ModelParams based on the dataset;
- -r n: Downsample the images by a factor of n to accelerate the training speed;
- --max_abs_split_points 0: For weakly textured scenes, to prevent overfitting in areas with weak textures, we recommend disabling this splitting strategy by setting it to 0;
- --opacity_cull_threshold 0.05: To reduce the number of Gaussian point clouds in a simple way, you can set this threshold.
```shell
# Training
python train.py -s data_path -m out_path --max_abs_split_points 0 --opacity_cull_threshold 0.05
```

#### Some Suggestions:
- Adjust max_depth and voxel_size based on the dataset;
- --use_depth_filter: Enable depth filtering to remove potentially inaccurate depth points using single-view and multi-view techniques. For scenes with floating points or insufficient viewpoints, it is recommended to turn this on.
```shell
# Rendering and Extract Mesh
python render.py -m out_path --max_depth 10.0 --voxel_size 0.01
```

## Acknowledgements
This project is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). Densify is based on [AbsGau](https://ty424.github.io/AbsGS.github.io/) and [GOF](https://github.com/autonomousvision/gaussian-opacity-fields?tab=readme-ov-file). DTU and Tanks and Temples dataset preprocess are based on [Neuralangelo scripts](https://github.com/NVlabs/neuralangelo/blob/main/DATA_PROCESSING.md). Evaluation scripts for DTU and Tanks and Temples dataset are based on [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation) respectively. We thank all the authors for their great work and repos. 


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{chen2024pgsr,
  title={PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction},
  author={Chen, Danpeng and Li, Hai and Ye, Weicai and Wang, Yifan and Xie, Weijian and Zhai, Shangjin and Wang, Nan and Liu, Haomin and Bao, Hujun and Zhang, Guofeng},
  journal={arXiv preprint arXiv:2406.06521},
  year={2024}
}
```
