# MinkLoc3D: Point Cloud Based Large-Scale Place Recognition
## MinkLoc3Dv2 is an improved version of our earlier point cloud descriptor MinkLoc3D. MinkLoc3Dv2 outperforms SOTA on standard benchmarks (as per February 2022).  

Paper: [Improving Point Cloud Based Place Recognition with Ranking-based Loss and Large Batch Training](https://xxxxx 
Submitted to: 2022 International Conference on Pattern Recognition (ICPR)
[Jacek Komorowski](mailto:jacek.komorowski@pw.edu.pl)
Warsaw University of Technology

### What's new ###
* [2022-02-01] Evaluation code and trained model of MinkLoc3Dv2 is released. 

### Our other projects ###
* MinkLoc3D: Point Cloud Based Large-Scale Place Recognition (WACV 2021): [MinkLoc3D](https://github.com/jac99/MinkLoc3D)
* MinkLoc++: Lidar and Monocular Image Fusion for Place Recognition (IJCNN 2021): [MinkLoc++](https://github.com/jac99/MinkLocMultimodal)
* Large-Scale Topological Radar Localization Using Learned Descriptors (ICONIP 2021): [RadarLoc](https://github.com/jac99/RadarLoc)
* EgonNN: Egocentric Neural Network for Point Cloud Based 6DoF Relocalization at the City Scale (IEEE Robotics and Automation Letters April 2022): [EgoNN](https://github.com/jac99/Egonn) 

### Introduction
The paper presents a simple and effective learning-based method for computing a discriminative 3D point cloud descriptor for place recognition purposes. 
Recent state-of-the-art methods have relatively complex architectures such as multi-scale pyramid of point Transformers combined with a pyramid of feature aggregation modules.
Our method uses a simple and efficient 3D convolutional feature extraction, based on a sparse voxelized representation, enhanced with channel attention blocks. 
We employ recent advances in image retrieval and propose a modified version of a loss function based on a differentiable average precision approximation. Such loss function requires training with very large batches for the best results. This is enabled by using multistaged backpropagation.
Experimental evaluation on the popular benchmarks proves the effectiveness of our approach, with a consistent improvement over state of the art.
![Overview](media/overview.jpg)

### Citation
If you find this work useful, please consider citing:

TO BE COMPLETED

### Environment and Dependencies
Code was tested using Python 3.8 with PyTorch 1.10.1 and MinkowskiEngine 0.5.4 on Ubuntu 20.04 with CUDA 10.2.
Note: CUDA 11.1 is not recommended as there are some issues with MinkowskiEngine 0.5.4 on CUDA 11.1. 

The following Python packages are required:
* PyTorch (version 1.10.1)
* MinkowskiEngine (version 0.5.4)
* pytorch_metric_learning (version 1.1 or above)
* wandb

Modify the `PYTHONPATH` environment variable to include absolute path to the project root folder: 
```export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/.../MinkLoc3D
```

### Datasets

**MinkLoc3Dv2** is trained on a subset of Oxford RobotCar and In-house (U.S., R.A., B.D.) datasets introduced in
*PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition* paper ([link](https://arxiv.org/pdf/1804.03492)).
There are two training datasets:
- Baseline Dataset - consists of a training subset of Oxford RobotCar
- Refined Dataset - consists of training subset of Oxford RobotCar and training subset of In-house

For dataset description see PointNetVLAD paper or github repository ([link](https://github.com/mikacuy/pointnetvlad)).

You can download training and evaluation datasets from 
[here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q) 
([alternative link](https://drive.google.com/file/d/1-1HA9Etw2PpZ8zHd3cjrfiZa8xzbp41J/view?usp=sharing)). 

Before the network training or evaluation, run the below code to generate pickles with positive and negative point clouds for each anchor point cloud. 

```generate pickles
cd generating_queries/ 

# Generate training tuples for the Baseline Dataset
python generate_training_tuples_baseline.py --dataset_root <dataset_root_path>

# Generate training tuples for the Refined Dataset
python generate_training_tuples_refine.py --dataset_root <dataset_root_path>

# Generate evaluation tuples
python generate_test_sets.py --dataset_root <dataset_root_path>
```
`<dataset_root_path>` is a path to dataset root folder, e.g. `/data/pointnetvlad/benchmark_datasets/`.
Before running the code, ensure you have read/write rights to `<dataset_root_path>`, as training and evaluation pickles
are saved there. 

### Training
Training code will be released after the paper acceptance.

### Pre-trained Models

Pretrained models are available in `weights` directory
- `minkloc3dv2_baseline.pth` trained on the Baseline Dataset 
- `minkloc3dv2_refined.pth` trained on the Refined Dataset 

### Evaluation

To evaluate pretrained models run the following commands:

```eval baseline
cd eval

# To evaluate the model trained on the Baseline Dataset
python evaluate.py --config ../config/config_baseline.txt --model_config ../models/minkloc3dv2.txt --weights ../weights/minkloc3dv2_baseline.pth

# To evaluate the model trained on the Refined Dataset
python evaluate.py --config ../config/config_refined.txt --model_config ../models/minkloc3dv2.txt --weights ../weights/minkloc3dv2_refined.pth
```

## Results

**MinkLoc3D** performance (measured by Average Recall@1) compared to state-of-the-art:

### Trained on Baseline Dataset

| Method                   | Oxford     | U.S.       | R.A.       | B.D        | Average    |
|--------------------------|------------|------------|------------|------------|------------|
| PointNetVLAD [1]         | 62.8       | 63.2       | 56.1       | 57.2       | 59.8       |
| PCAN [2]                 | 69.1       | 62.4       | 56.9       | 58.1       | 61.6       |
| LPD-Net [4]              | 86.3       | 87.0       | 83.1       | 82.5       | 94.7       |
| EPC-Net [5]              | 86.2       | -          | -          | -          | -          | 
| NDT-Transformer [7]      | 93.8       | -          | -          | -          | -          |
| MinkLoc3D [8]            | 93.0       | 86.7       | 80.4       | 81.5       | 85.4       |
| PPT-Net [9]              | 93.5       | 90.1       | 84.1       | 84.6       | 88.1       |
| SVT-Net [10]             | 93.7       | 90.1       | 84.4       | 85.5       | 88.4       |
| TransLoc3D [11]          | 95.0       | -          | -          | -          | -          |
| ***MinkLoc3Dv2 (ours)*** | ***96.3*** | ***90.9*** | ***86.5*** | ***86.3*** | ***90.0*** |


### Trained on Refined Dataset


| Method                   | Oxford     | U.S.       | R.A.       | B.D         | Average    |
|--------------------------|------------|------------|------------|-------------|------------|
| PointNetVLAD [1]         | 63.3       | 86.1       | 82.7       | 80.1        | 78.0       |
| PCAN [2]                 | 70.7       | 83.7       | 82.5       | 80.3        | 79.3       |
| DAGC [3]                 | 71.5       | 86.3       | 82.8       | 81.3        | 80.5       |
| LPD-Net [4]              | 86.6       | 94.4       | 90.8       | 90.8        | 90.7       |
| SOE-Net [6]              | 89.3       | 91.8       | 90.2       | 89.0        | 90.1       | 
| MinkLoc3D [8]            | 94.8       | 97.2       | 96.7       | 94.0        | 95.7       |
| SVT-Net [10]             | 93.7       | 97.0       | 95.2       | 94.4        | 95.3       |
| TransLoc3D [11]          | 95.0       | 97.5       | 97.3       | 94.8        | 96.2       |
| ***MinkLoc3Dv2 (ours)*** | ***96.9*** | ***99.0*** | ***98.3*** | ***97.6***  | ***97.9*** |

1. M. A. Uy and G. H. Lee, "PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition", 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
2. W. Zhang and C. Xiao, "PCAN: 3D Attention Map Learning Using Contextual Information for Point Cloud Based Retrieval", 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
3. Q. Sun et al., "DAGC: Employing Dual Attention and Graph Convolution for Point Cloud based Place Recognition", 2020 International Conference on Multimedia Retrieval
4. Z. Liu et al., "LPD-Net: 3D Point Cloud Learning for Large-Scale Place Recognition and Environment Analysis", 2019 IEEE/CVF International Conference on Computer Vision (ICCV)
5. L. Hui et al., "Efficient 3D Point Cloud Feature Learning for Large-Scale Place Recognition", preprint arXiv:2101.02374 (2021)
6. Y. Xia et al., "SOE-Net: A Self-Attention and Orientation Encoding Network for Point Cloud based Place Recognition", 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
7. Z. Zhou et al., "NDT-Transformer: Large-scale 3D Point Cloud Localisation Using the Normal Distribution Transform Representation", 
   2021 IEEE International Conference on Robotics and Automation (ICRA)
8. J. Komorowski, "MinkLoc3D: Point Cloud Based Large-Scale Place Recognition", 2021 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)
9. L. Hui et al.,"Pyramid Point Cloud Transformer for Large-Scale Place Recognition", 2021 IEEE/CVF International Conference on Computer Vision
10. Z. Fan et al., "SVT-Net: Super lightweight Sparse Voxel Transformer for Large Scale Place Recognition", arXiv:2105.00149 (2021)
11. T. Xu et al., "TransLoc3d: Point Cloud Based Large-Scale Place Recognition using Adaptive Receptive Fields", arXiv:2105.11605 (2021)

### License
Our code is released under the MIT License (see LICENSE file for details).
