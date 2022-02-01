# Warsaw University of Technology

import numpy as np
import random
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sklearn.neighbors import KDTree

from datasets.base_datasets import TrainingDataset, EvaluationTuple
from datasets.mulran.mulran_train import MulranTrainingDataset
from datasets.southbay.southbay_train import SouthbayTrainingDataset
from datasets.pointnetvlad.pnv_train import PNVTrainingDataset
from datasets.augmentation import TrainTransform, TrainSetTransform, VirtualNegativeTransform
from datasets.pointnetvlad.pnv_train import TrainTransform as PNVTrainTransform
from datasets.samplers import BatchSampler
from misc.utils import TrainingParams
from datasets.base_datasets import PointCloudLoader
from datasets.mulran.mulran_raw import MulranPointCloudLoader
from datasets.southbay.southbay_raw import SouthbayPointCloudLoader
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader


def get_pointcloud_loader(dataset_type) -> PointCloudLoader:
    if dataset_type == 'mulran':
        return MulranPointCloudLoader()
    elif dataset_type == 'southbay':
        return SouthbayPointCloudLoader()
    elif dataset_type == 'kitti':
        return MulranPointCloudLoader()
    elif dataset_type == 'pnv':
        # PointNetVLAD datasets: based on Oxford RobotCar and Inline
        return PNVPointCloudLoader()
    else:
        raise NotImplementedError(f"Unsupported dataset type: {dataset_type}")


def make_datasets(params: TrainingParams, validation: bool = True):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.set_aug_mode)

    if params.dataset == "mulran":
        datasets['train'] = MulranTrainingDataset(params.dataset_folder, params.train_file,
                                                  transform=train_transform, set_transform=train_set_transform)
        if validation:
            datasets['val'] = MulranTrainingDataset(params.dataset_folder, params.val_file)
    elif params.dataset == "southbay":
        datasets['train'] = SouthbayTrainingDataset(params.dataset_folder, params.train_file,
                                                    transform=train_transform, set_transform=train_set_transform)
        if validation:
            datasets['val'] = SouthbayTrainingDataset(params.dataset_folder, params.val_file)
    elif params.dataset == "pnv":
        # PoinNetVLAD datasets (RobotCar and Inhouse)
        # PNV datasets have their own transform
        train_transform = PNVTrainTransform(params.aug_mode)
        datasets['train'] = PNVTrainingDataset(params.dataset_folder, params.train_file,
                                               transform=train_transform, set_transform=train_set_transform)
        if validation:
            datasets['val'] = PNVTrainingDataset(params.dataset_folder, params.val_file)
    else:
        raise NotImplementedError("Dataset not supported: {params.dataset}")

    if params.secondary_dataset is None:
        pass
    elif params.secondary_dataset == "mulran":
        datasets['secondary_train'] = MulranTrainingDataset(params.secondary_dataset_folder,
                                                            params.secondary_train_file,
                                                            transform=train_transform,
                                                            set_transform=train_set_transform)
        #datasets['secondary_val'] = MulranTrainingDataset(params.secondary_dataset_folder, params.val_file)
    elif params.secondary_dataset == "southbay":
        datasets['secondary_train'] = SouthbayTrainingDataset(params.secondary_dataset_folder,
                                                              params.secondary_train_file,
                                                              transform=train_transform,
                                                              set_transform=train_set_transform)
        #datasets['secondary_val'] = SouthbayTrainingDataset(params.secondary_dataset_folder, params.val_file)
    else:
        raise NotImplementedError("Dataset not supported: {params.dataset}")

    return datasets


def make_collate_fn(dataset: TrainingDataset, quantizer, batch_split_size=None,  virtual_negatives: bool = False):
    # quantizer: converts to polar (when polar coords are used) and quantizes
    # batch_split_size: if not None, splits the batch into a list of multiple mini-batches with batch_split_size elems
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]

        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            lens = [len(cloud) for cloud in clouds]
            clouds = torch.cat(clouds, dim=0)
            clouds = dataset.set_transform(clouds)
            clouds = clouds.split(lens)

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        if virtual_negatives:
            # Extend the batch with virtual negatives
            clouds, positives_mask, negatives_mask = generate_virtual_negatives(clouds, positives_mask, negatives_mask)

        # Convert to polar (when polar coords are used) and quantize
        # Use the first value returned by quantizer
        coords = [quantizer(e)[0] for e in clouds]

        if batch_split_size is None or batch_split_size == 0:
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': coords, 'features': feats}

        else:
            # Split the batch into chunks
            batch = []
            for i in range(0, len(coords), batch_split_size):
                temp = coords[i:i + batch_split_size]
                c = ME.utils.batched_coordinates(temp)
                f = torch.ones((c.shape[0], 1), dtype=torch.float32)
                minibatch = {'coords': c, 'features': f}
                batch.append(minibatch)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and negatives_mask which are
        # batch_size x batch_size boolean tensors
        #return batch, positives_mask, negatives_mask, torch.tensor(sampled_positive_ndx), torch.tensor(relative_poses)
        return batch, positives_mask, negatives_mask

    return collate_fn


def make_dataloaders(params: TrainingParams, validation=True):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, validation=validation)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    quantizer = params.model_params.quantizer
    train_collate_fn = make_collate_fn(datasets['train'],  quantizer, params.batch_split_size,
                                       virtual_negatives=params.virtual_negatives)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler,
                                     collate_fn=train_collate_fn, num_workers=params.num_workers,
                                     pin_memory=True)
    if validation and 'val' in datasets:
        val_collate_fn = make_collate_fn(datasets['val'], quantizer, params.batch_split_size, virtual_negatives=False)
        val_sampler = BatchSampler(datasets['val'], batch_size=params.val_batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)

    if params.secondary_dataset is not None:
        secondary_train_sampler = BatchSampler(datasets['secondary_train'], batch_size=params.batch_size,
                                               batch_size_limit=params.secondary_batch_size_limit,
                                               batch_expansion_rate=params.batch_expansion_rate, max_batches=2000)

        secondary_train_collate_fn = make_collate_fn(datasets['secondary_train'],  quantizer, params.batch_split_size,
                                                     virtual_negatives=params.virtual_negatives)
        dataloders['secondary_train'] = DataLoader(datasets['secondary_train'],
                                                   batch_sampler=secondary_train_sampler,
                                                   collate_fn=secondary_train_collate_fn,
                                                   num_workers=params.num_workers,
                                                   pin_memory=True)

    return dataloders


def filter_query_elements(query_set: List[EvaluationTuple], map_set: List[EvaluationTuple],
                          dist_threshold: float) -> List[EvaluationTuple]:
    # Function used in evaluation dataset generation
    # Filters out query elements without a corresponding map element within dist_threshold threshold
    map_pos = np.zeros((len(map_set), 2), dtype=np.float32)
    for ndx, e in enumerate(map_set):
        map_pos[ndx] = e.position

    # Build a kdtree
    kdtree = KDTree(map_pos)

    filtered_query_set = []
    count_ignored = 0
    for ndx, e in enumerate(query_set):
        position = e.position.reshape(1, -1)
        nn = kdtree.query_radius(position, dist_threshold, count_only=True)[0]
        if nn > 0:
            filtered_query_set.append(e)
        else:
            count_ignored += 1

    print(f"{count_ignored} query elements ignored - not having corresponding map element within {dist_threshold} [m] "
          f"radius")
    return filtered_query_set


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e


def generate_virtual_negatives(clouds: Tuple[torch.Tensor], positives_mask: torch.Tensor,
                               negatives_mask: torch.Tensor):
    # Sample one (first in the row) positive for each embedding

    batch_size = len(clouds)

    # Generate app. 50% of batch size of virtual negatives
    n_virtual_examples = batch_size // 2
    samples = random.sample(range(batch_size), k=n_virtual_examples)

    # Create 'virtual negatives' using selected examples

    # Generate virtual negatives
    # If point cloud N is negative to A, then "virtual negatives" generated from N are also negatives to A
    t = VirtualNegativeTransform()
    virtual_clouds = [t(clouds[i]) for i in samples]

    new_clouds = list(clouds)
    new_clouds.extend(virtual_clouds)

    # Virtual clouds have all False positive mask (they are used only as negatives)
    new_positives_mask = torch.full((batch_size + n_virtual_examples, batch_size + n_virtual_examples),
                                    fill_value=False, dtype=torch.bool)
    new_positives_mask[:batch_size, :batch_size] = positives_mask

    new_negatives_mask = torch.full((batch_size + n_virtual_examples, batch_size + n_virtual_examples), fill_value=False, dtype=torch.bool)
    new_negatives_mask[:batch_size, :batch_size] = negatives_mask

    # 'virtual negative' have the same negative masks as their base examples
    virtual_negatives_mask = negatives_mask[samples]

    # After transposition, rows correspond to original point clouds, columns correspond to new virtual negatives
    new_negatives_mask[:batch_size, batch_size:] = virtual_negatives_mask.t()

    return new_clouds, new_positives_mask, new_negatives_mask