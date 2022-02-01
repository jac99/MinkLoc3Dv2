# Base dataset classes, inherited by dataset-specific classes
import os
import pickle
from typing import List
import numpy as np


class EvaluationTuple:
    # Tuple describing an evaluation set element
    def __init__(self, timestamp: int, rel_scan_filepath: str, position: np.array):
        # position: x, y position in meters
        assert position.shape == (2,)
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.position = position

    def to_tuple(self):
        return self.timestamp, self.rel_scan_filepath, self.position


class EvaluationSet:
    # Evaluation set consisting of map and query elements
    def __init__(self, query_set: List[EvaluationTuple] = None, map_set: List[EvaluationTuple] = None):
        self.query_set = query_set
        self.map_set = map_set

    def save(self, pickle_filepath: str):
        # Pickle the evaluation set

        # Convert data to tuples and save as tuples
        query_l = []
        for e in self.query_set:
            query_l.append(e.to_tuple())

        map_l = []
        for e in self.map_set:
            map_l.append(e.to_tuple())
        pickle.dump([query_l, map_l], open(pickle_filepath, 'wb'))

    def load(self, pickle_filepath: str):
        # Load evaluation set from the pickle
        query_l, map_l = pickle.load(open(pickle_filepath, 'rb'))

        self.query_set = []
        for e in query_l:
            self.query_set.append(EvaluationTuple(e[0], e[1], e[2]))

        self.map_set = []
        for e in map_l:
            self.map_set.append(EvaluationTuple(e[0], e[1], e[2]))

    def get_map_positions(self):
        # Get map positions as (N, 2) array
        positions = np.zeros((len(self.map_set), 2), dtype=self.map_set[0].position.dtype)
        for ndx, pos in enumerate(self.map_set):
            positions[ndx] = pos.position
        return positions

    def get_query_positions(self):
        # Get query positions as (N, 2) array
        positions = np.zeros((len(self.query_set), 2), dtype=self.query_set[0].position.dtype)
        for ndx, pos in enumerate(self.query_set):
            positions[ndx] = pos.position
        return positions


class PointCloudLoader:
    def __init__(self):
        # remove_zero_points: remove points with all zero coordinates
        # remove_ground_plane: remove points on ground plane level and below
        # ground_plane_level: ground plane level
        self.remove_zero_points = True
        self.remove_ground_plane = True
        self.ground_plane_level = None
        self.set_properties()

    def set_properties(self):
        # Set point cloud properties, such as ground_plane_level. Must be defined in inherited classes.
        raise NotImplementedError('set_properties must be defined in inherited classes')

    def __call__(self, file_pathname):
        # Reads the point cloud from a disk and preprocess (optional removal of zero points and points on the ground
        # plane and below
        # file_pathname: relative file path
        assert os.path.exists(file_pathname), f"Cannot open point cloud: {file_pathname}"
        pc = self.read_pc(file_pathname)
        assert pc.shape[1] == 3

        if self.remove_zero_points:
            mask = np.all(np.isclose(pc, 0), axis=1)
            pc = pc[~mask]

        if self.remove_ground_plane:
            mask = pc[:, 2] > self.ground_plane_level
            pc = pc[mask]

        return pc

    def read_pc(self, file_pathname: str) -> np.ndarray:
        # Reads the point cloud without pre-processing
        raise NotImplementedError("read_pc must be overloaded in an inheriting class")
