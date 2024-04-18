<h1 align="center">CT-MVSNet: Efficient Multi-View Stereo with Cross-scale Transformer(MMM‘24 Oral)</h1>

<div align="center">
    <a href="https://github.com/wscstrive" target='_blank'>Sicheng Wang</a>, 
    <a href="https://scholar.google.com/citations?hl=en&user=wv7uLCMAAAAJ" target='_blank'>Hao Jiang</a>*, 
    <a href="https://github.com/Sion1" target='_blank'>Lei Xiang</a>
</div>

<br />

<div align="center">
    <a href="https://link.springer.com/chapter/10.1007/978-3-031-53308-2_29" target='_blank'><img src="https://img.shields.io/badge/MMM-2024-1E90FF"/></a>&nbsp;
    <a href="https://arxiv.org/abs/2312.08594.pdf" target='_blank'><img src="https://img.shields.io/badge/arxiv-arxiv.2312.08594-B31B1B?logo=arXiv&logoColor=green"/></a>&nbsp;
</div>




## 📌 Introduction
In this paper, we propose a novel cross-scale transformer (CT) that processes feature representations at different stages without additional computation. Specifically, we introduce an adaptive matching-aware transformer (AMT) that employs different interactive attention combinations at multiple scales. This combined strategy enables our network to capture intra-image context information and enhance inter-image feature relationships. Besides, we present a dual-feature guided aggregation (DFGA) that embeds the coarse global semantic information into the finer cost volume construction to further strengthen global and local feature awareness. Meanwhile, we design a feature metric loss (FM Loss) that evaluates the feature bias before and after transformation to reduce the impact of feature mismatch on depth estimation. Extensive experiments on DTU dataset and Tanks and Temples benchmark demonstrate that our method achieves state-of-the-art results.
![](asserts/overview.png)


## 🌑 Preparation

### ✔ Repo & Environment

Our code is tested with Python==3.8, PyTorch==1.9.0,  CUDA==10.2 on Ubuntu-18.04 with NVIDIA GeForce RTX 2080Ti.

To use CT-MVSNet, clone this repo:
```
git clone https://github.com/wscstrive/CT-MVSNet.git
cd CT-MVSNet
```
Use the following commands to build the `conda` environment.
```
conda create -n ctmvsnet python=3.6
conda activate ctmvsnet
pip install -r requirements.txt
```

### ✔ Datasets
In TransMVSNet, we mainly use [DTU](https://roboimagedata.compute.dtu.dk/), [BlendedMVS](https://github.com/YoYo000/BlendedMVS/) and [Tanks and Temples](https://www.tanksandtemples.org/) to train and evaluate our models. You can prepare the corresponding data by following the instructions below.

#### DTU Dataset
For DTU training set, you can download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view)
 and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip)
 (both from [Original MVSNet](https://github.com/YoYo000/MVSNet)), and unzip them to construct a dataset folder like:
```
dtu_training
 ├── Cameras
 ├── Depths
 ├── Depths_raw
 └── Rectified
```
For DTU testing set, you can download the preprocessed [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet)) and unzip it as the test data folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.

#### BlendedMVS Dataset
We use the [low-res set](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV) of BlendedMVS dataset for both training and testing. You can download the [low-res set](https://1drv.ms/u/s!Ag8Dbz2Aqc81gVDgxb8MDGgoV74S?e=hJKlvV) from [orignal BlendedMVS](https://github.com/YoYo000/BlendedMVS) and unzip it to form the dataset folder like below:
```
BlendedMVS
 ├── 5a0271884e62597cdee0d0eb
 │     ├── blended_images
 │     ├── cams
 │     └── rendered_depth_maps
 ├── 59338e76772c3e6384afbb15
 ├── 59f363a8b45be22330016cad
 ├── ...
 ├── all_list.txt
 ├── training_list.txt
 └── validation_list.txt
```

#### Tanks and Temples Dataset
Download our preprocessed [Tanks and Temples dataset](https://drive.google.com/file/d/1IHG5GCJK1pDVhDtTHFS3sY-ePaK75Qzg/view?usp=sharing) and unzip it to form the dataset folder like below:
```
tankandtemples
 ├── advanced
 │  ├── Auditorium
 │  ├── Ballroom
 │  ├── ...
 │  └── Temple
 └── intermediate
        ├── Family
        ├── Francis
        ├── ...
        └── Train
```

## 🌒 Training

### ✔ Training on DTU
Set the configuration in ``scripts/train.sh``:
* Set ``MVS_TRAINING`` as the path of DTU training set.
* Set ``LOG_DIR`` to save the checkpoints.
* Change ``NGPUS`` to suit your device.
* We use ``torch.distributed.launch`` by default.

To train your  own model, just run:
```
bash scripts/train.sh
```
You can conveniently modify more hyper-parameters in ``scripts/train.sh`` according to the argparser in ``train.py``, such as ``summary_freq``, ``save_freq``, and so on.

### ✔ Finetune on BlendedMVS
For a fair comparison with other SOTA methods on Tanks and Temples benchmark, we finetune our model on BlendedMVS dataset after training on DTU dataset.

Set the configuration in ``scripts/train_bld_fintune.sh``:
* Set ``MVS_TRAINING`` as the path of BlendedMVS dataset.
* Set ``LOG_DIR`` to save the checkpoints and training log.
* Set ``CKPT`` as path of the loaded ``.ckpt`` which is trained on DTU dataset.

To finetune your own model, just run:
```
bash scripts/train_bld_fintune.sh
```

## 🌓 Testing

### ✔ Testing on DTU

**Important Tips:** to reproduce our reported results, you need to:
* compile and install the modified `gipuma` from [Yao Yao](https://github.com/YoYo000/fusibile) as introduced below
* use the latest code as we have fixed tiny bugs and updated the fusion parameters
* make sure you install the right version of python and pytorch, use some old versions would throw warnings of the default action of `align_corner` in several functions, which would affect the final results
* be aware that we only test the code on 2080Ti and Ubuntu 18.04, other devices and systems might get slightly different results
* make sure that you use the `*.ckpt` for testing


To start testing, set the configuration in ``scripts/test_dtu.sh``:
* Set ``TESTPATH`` as the path of DTU testing set.
* Set ``TESTLIST`` as the path of test list (.txt file).
* Set ``CKPT_FILE`` as the path of the model weights.
* Set ``OUTDIR`` as the path to save results.

Run:
```
bash scripts/test_dtu.sh
```


<!-- The simple instruction for installing and compiling `gipuma` can be found [here](https://github.com/YoYo000/MVSNet#post-processing).  The installed gipuma is a modified version from [Yao Yao](https://github.com/YoYo000/fusibile).-->
To install the `gipuma`, clone the modified version from [Yao Yao](https://github.com/YoYo000/fusibile).
Modify the line-10 in `CMakeLists.txt` to suit your GPUs. Othervise you would meet warnings when compile it, which would lead to failure and get 0 points in fused point cloud. For example, if you use 2080Ti GPU, modify the line-10 to:
```
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --use_fast_math --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_70,code=sm_70)
```
If you use other kind of GPUs, please modify the arch code to suit your device (`arch=compute_XX,code=sm_XX`).
Then install it by `cmake .` and `make`, which will generate the executable file at `FUSIBILE_EXE_PATH`.
Please note 



For quantitative evaluation on DTU dataset, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36). Unzip them and place `Points` folder in `SampleSet/MVS Data/`. The structure looks like:
```
SampleSet
├──MVS Data
      └──Points
```
In ``DTU-MATLAB/BaseEvalMain_web.m``, set `dataPath` as path to `SampleSet/MVS Data/`, `plyPath` as directory that stores the reconstructed point clouds and `resultsPath` as directory to store the evaluation results. Then run ``DTU-MATLAB/BaseEvalMain_web.m`` in matlab.

| DTU Dataset | Acc. ↓       | Comp. ↓        | Overall ↓ |
|-------------|--------------|----------------|-----------|
| CT-MVSNet   | 0.341        | 0.264          | 0.302     |




### ✔ Testing on Tanks and Temples
We recommend using the finetuned models `*.ckpt` to test on Tanks and Temples benchmark.

Similarly, set the configuration in ``scripts/test_tnt.sh``:
* Set ``TESTPATH`` as the path of intermediate set or advanced set.
* Set ``TESTLIST`` as the path of test list (.txt file).
* Set ``CKPT_FILE`` as the path of the model weights.
* Set ``OUTDIR`` as the path to save resutls.

To generate point cloud results, just run:
```
bash scripts/test_tnt.sh
```
Note that：
* The parameters of point cloud fusion have not been studied thoroughly and the performance can be better if cherry-picking more appropriate thresholds for each of the scenes.
* The dynamic fusion code is borrowed from [AA-RMVSNet](https://github.com/QT-Zhu/AA-RMVSNet).

For quantitative evaluation, you can upload your point clouds to [Tanks and Temples benchmark](https://www.tanksandtemples.org/).

| T&T (Intermediate) | Mean ↑ | Family | Francis | Horse | Lighthouse | M60   | Panther | Playground | Train |
|--------------------|--------|--------|---------|-------|------------|-------|---------|------------|-------|
| CT-MVSNet          | 64.28  | 81.20  | 65.09   | 56.95 | 62.60      | 63.07 | 64.83   | 61.82      | 58.68 |

| T&T (Advanced) | Mean ↑ | Auditorium | Ballroom | Courtroom | Museum | Palace | Temple |
|----------------|--------|------------|----------|-----------|--------|--------|--------|
| CT-MVSNet      | 38.03  | 28.37      | 44.61    | 34.83     | 46.51  | 34.69  | 39.15  |
## 🔗 Citation

```bibtex
@inproceedings{wang2024ct,
  title={CT-MVSNet: Efficient Multi-view Stereo with Cross-Scale Transformer},
  author={Wang, Sicheng and Jiang, Hao and Xiang, Lei},
  booktitle={International Conference on Multimedia Modeling},
  pages={394--408},
  year={2024},
  organization={Springer}
}
```

## 💌 Acknowledgments
We borrow some code from [CasMVSNet](https://github.com/alibaba/cascade-stereo/tree/master/CasMVSNet), [TransMVSNet](https://github.com/megvii-research/TransMVSNet). We thank the authors for releasing the source code.
