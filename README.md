
<h1> Rearrangement Planning for General Part Assembly </h1>

![teaser](assets/gpat_teaser.jpg)

[Yulong Li](https://www.columbia.edu/~yl4095/)<sup>1</sup>,
[Andy Zeng](https://andyzeng.github.io)<sup>2</sup>,
[Shuran Song](https://www.cs.columbia.edu/~shurans/)<sup>1</sup>
<br>
<sup>1</sup>Columbia University, <sup>2</sup> Google Deepmind
<br>

[Project Page](https://general-part-assembly.github.io) | [arXiV](https://arxiv.org/abs/2307.00206)

<br>

## Catalog
- [Catalog](#catalog)
- [Environment](#environment)
- [Dataset](#dataset)
- [Training and Testing](#training-and-testing)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
  
## Environment
```
conda env create -f environment.yml
conda activate gpat
pip install -e .
```
Install other dependencies:
- [chamfer](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly/tree/main/exps/utils/chamfer)
```
cd utils/chamfer; python setup.py install
```
- [pointops](https://github.com/POSTECH-CVLab/point-transformer/tree/master/lib/pointops)
```
cd utils/pointops; python setup.py install
```

## Dataset
We create our dataset from [PartNet](https://partnet.cs.stanford.edu).
Download our [processed partnet dataset]() to `dataset/partnet`.

To recreate our data, first download PartNet v0 (annotations, meshes, point clouds, and visualizations) to `dataset/partnet_raw`. Also download [PartNet meta data](https://github.com/daerduoCarey/partnet_dataset.git) to `dataset/partnet_dataset`. Then run the following command:
```python3
python dataset/preprocess.py
```

For custom datasets, each data point should have the following files in its folder:
```bash
target.npy # (5000, 3), required, target point-cloud.
parts.npy # (K, 1000, 3), required, parts point-clouds.
target_100k.npy # (100000, 3), optional, a dense target point-cloud. Optionally include this for better assembly results.
target_100k_labels.npy # (10000), optional, indicates the nearest neighbor of target_100k.npy in target.npy, with values from [0, 5000). Optionally include this for better assembly results.
poses.npy # (K, 7), optional, GT poses for each part. First three coordinates denote (x, y, z) position, last four coordinates denote a quaternion with real-part first. Optionally include this for correct evaluation.
labels.npy # (5000), optional, GT segmentation label of the target, each index takes a value from [0, K). Optionally include this for correct evaluation of segmentation.
eq_class.npy # (K), optional, equivalence classes of the parts. For example, [0,0,1,2,2,2] means that the first two parts are equivalent, and last three parts are equivalent.  Optionally include this for correct evaluation.
```
## Training and Testing
To re-train GPAT,
```python3
python learning/gpat/run.py --mode=train --ratio=0.7 --rand --exp=EXPNAME --cuda=CUDAIND
```
To evaluate with our pretrained GPAT checkpoint, download the [checkpoint](https://drive.google.com/file/d/1tdoUWV39MtJ09_Lk6lSKCBgFa5mGNVHk/view?usp=share_link) to `logs/pretrained/gpat.pth`.

```python3
python learning/assembler.py --eval --cat=CATEGORY --exp=EXPNAME --cuda=CUDAIND
```
Pass in `--ratio=1` to test non-exact parts only (or another ratio from [0, 1] to indicate the probability of testing with non-exact parts). Pass in `--rand` to test targets at random poses. To evaluate with custom dataset, specify the directory containing all the data folders ([instructions](#dataset)) with `--eval_dir` (note that you cannot directly point to a single data folder). For custom model checkpoints, pass in the model path with `--model_path`. For other parameters, see `init_args()` in [learning_utils.py](learning/learning_utils.py). 

## Acknowledgements
- [PartNet](https://partnet.cs.stanford.edu)
- [Generative 3D Part Assembly via Dynamic Graph Learning](https://github.com/hyperplane-lab/Generative-3D-Part-Assembly)
- [PointTransformer](https://github.com/POSTECH-CVLab/point-transformer)
- [Zhenjia Xu](https://www.zhenjiaxu.com/): [Html-Visualization](https://github.com/columbia-ai-robotics/html-visualization)

## Citation
If you find this codebase useful, feel free to cite our work!
<div style="display:flex;">
<div>

```bibtex
@inproceedings{li2023rearrangement,
	title={Rearrangement Planning for General Part Assembly},
	author={Li, Yulong and Zeng, Andy and Song, Shuran},
	booktitle={Conference on Robot Learning},
	year={2023},
	organization={PMLR}
}
```
