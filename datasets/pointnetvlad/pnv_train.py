# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# For information on dataset see: https://github.com/mikacuy/pointnetvlad
# Warsaw University of Technology

import numpy as np
import torchvision.transforms as transforms

from datasets.augmentation import JitterPoints, RemoveRandomPoints, RandomTranslation, RandomRotation, \
    RemoveRandomBlock, RotationByAngle
from datasets.base_datasets import TrainingDataset
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader


class PNVTrainingDataset(TrainingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pc_loader = PNVPointCloudLoader()


class TrainTransform:
    # Augmentations specific for PointNetVLAD datasets (RobotCar and Inhouse)
    def __init__(self, aug_mode):
        self.aug_mode = aug_mode
        if self.aug_mode == 1:
            # Augmentations without random rotation around z-axis
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]
        elif self.aug_mode == 2:
            # Augmentations with random rotation around z-axis
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RandomTranslation(max_delta=0.01),
                 RandomRotation(max_theta=180, axis=np.array([0, 0, 1])),
                 RemoveRandomBlock(p=0.4)]
        elif self.aug_mode == 3:
            # Augmentations without random rotation around z-axis
            t = [JitterPoints(sigma=0.001, clip=0.002), RemoveRandomPoints(r=(0.0, 0.1)),
                 RotationByAngle(thetas=[0., 180.]), RandomTranslation(max_delta=0.01), RemoveRandomBlock(p=0.4)]
        else:
            raise NotImplementedError('Unknown aug_mode: {}'.format(self.aug_mode))
        self.transform = transforms.Compose(t)

    def __call__(self, e):
        if self.transform is not None:
            e = self.transform(e)
        return e

