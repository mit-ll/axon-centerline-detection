import os
import h5py
import warnings
import numpy as np
import itertools
import pickle
import torch
from torch.utils import data
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ThreadPoolExecutor
from utils.transforms import GrayscaleAugmentation, AffineTransformation, Skeletonize
from utils.graph_skel import Graph


class AxonDataset(data.Dataset):
    '''
    Implements __len__ and __getitem__ methods for compatibility with PyTorch's DataLoader class.
    Creates a map-style dataset by sampling from a 3D image and its corresponding ground truth
    segmentation mask. During training (i.e. testing=False), samples are cropped and (optionally)
    augmented randomly. During testing, samples are drawn using a deterministic sliding window
    and a fixed set of augmentations.
    '''

    def __init__(self, image, truth, crop_size=None, augmentation=True,
                 include_centerline=False, testing=False, overlap=False,
                 **kwargs):
        '''
        Inputs:
          - `image`(numpy.ndarray): A 3- or 4-dimensional image (where the first dimension is channels)
          - `truth`(numpy.ndarray): Ground truth segmentation mask of the same shape as image
          - `crop_size`(iterable): Spatial dimensions of samples to be cropped from full volume
          - `augmentation`(bool): If True, will apply the augmentations defined in self.get_transforms()
          - `include_centerline`(bool): If True, will include a third output representing the ground truth centerline
          - `testing`(bool): If True, use deterministic testing behavior rather than random training behavior
          - `overlap`(bool): If True, uses a 50% overlap for test samples to reduce edge noise
        '''

        # Validate dimensions
        assert image.ndim == truth.ndim
        assert image.shape[-3:] == truth.shape[-3:]
        self.shape = np.array(image.shape[-3:])
        self.image = image
        self.truth = truth
        # Data attributes to be transformed and returned in self.__getitem__()
        self.data_keys = ['image', 'truth']
        # List attributes that are indexed by sample
        self.sample_keys = ['sample_indices']

        # Sample dimensions should be (channels, Z, Y, X)
        while self.image.ndim < 4:
            self.image = np.expand_dims(image, 0)
            self.truth = np.expand_dims(truth, 0)

        # Set spatial crop size
        if crop_size is not None:
            assert len(crop_size) == 3
            self.crop_size = np.array(crop_size)
        else:
            self.crop_size = self.shape

        # Include skeletonized truth as auxiliary output
        if include_centerline:
            self.centerline = Skeletonize(threshold=0.5).apply(self.truth)
            self.data_keys.append('centerline')

        # Preprocess data, choose samples, etc.
        self.set_context(testing=testing,
                         augmentation=augmentation,
                         overlap=overlap)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        (x1, y1, z1), (x2, y2, z2) = self.get_crop_coordinates(idx)

        data = [getattr(self, key)[..., x1:x2, y1:y2, z1:z2] for key in self.data_keys]

        if self.augmentation:
            data = self.transform(data, self.get_transforms(idx))

        return (idx, *data)

    def _get_sample_indices(self):
        '''
        Return a list of coordinates (i.e. lower bound in each dimension) of each sample.
        Note: Only used for testing, when cropping is deterministic.
        '''
        if self.overlap:
            step_size = self.crop_size//2
            num_samples = self.shape//step_size - 1
        else:
            step_size = self.crop_size
            num_samples = self.shape//step_size

        return list(np.ndindex(*num_samples))*step_size

    def _get_test_time_augmentations(self):
        '''
        Return a list of lists, each representing a unique set of transformations
        to use for test time augmentation.

        Includes every unique combination of flips along each axis as well as
        90 degree XY rotations for a total of 16 transformations.
        '''
        return [[AffineTransformation(flip=[x, y, z], angle=rotate*90)]
                for x, y, z, rotate
                in list(itertools.product([0, 1], repeat=4))]

    def _get_training_augmentations(self):
        '''
        Return a list of transformations to use at training time.

        Includes random grayscale augmentation, flips, and 90-degree XY rotations.
        '''
        return [GrayscaleAugmentation(random=True),
                AffineTransformation(rotation_step=90, random=True)]

    def _preprocess_data(self):
        '''
        Apply in place any necessary preprocessing to input dataset.
        '''
        if self.testing:
            for key in self.data_keys:
                data = getattr(self, key)
                # Undo any previous padding to ensure idempotency
                if hasattr(self, 'pads'):
                    data = self.unpad(data)
                # Pad to accomodate sliding window
                data, pads = self.pad(data)
                setattr(self, key, data)
            self.pads = np.array(pads)
            self.shape = np.array(self.image.shape[-3:])

    def get_crop_coordinates(self, idx):
        '''
        Return lower and upper bound coordinates of sample crop for given index.
        '''
        if self.testing:
            # Crop test data using sliding window
            lower = self.sample_indices[idx]
        else:
            # Crop training data randomly
            lower = [torch.randint(bound, (1,)).item()
                     for bound in self.shape-self.crop_size+1]
        upper = lower + self.crop_size
        return lower, upper

    def get_transforms(self, idx):
        # Test time transformations are deterministic
        if self.testing:
            transform_idx = idx % len(self.test_time_augmentations)
            return self.test_time_augmentations[transform_idx]
        # Training augmentation is done randomly
        return self._get_training_augmentations()

    def set_context(self, testing, augmentation, **kwargs):
        ''' Set variables that depend on testing context '''
        self.testing = testing
        self.augmentation = augmentation
        self.__dict__.update(kwargs)
        self._preprocess_data()

        # Get coordinates of samples
        self.sample_indices = self._get_sample_indices()
        # Repeat samples once for each test time augmentation
        if self.augmentation:
            self.test_time_augmentations = self._get_test_time_augmentations()
            for key in self.sample_keys:
                value = np.repeat(getattr(self, key),
                                  len(self.test_time_augmentations),
                                  axis=0)
                setattr(self, key, value)
        # Dataset length is determined by number of samples
        self.length = len(self.sample_indices)

    def transform(self, images, transforms, reverse=False):
        '''
        Apply transformations to each input. First image expected to be raw imagery.
        '''
        for transform in transforms:
            for i, image in enumerate(images):
                # Only spatial transforms can be applied to truth data
                if i == 0 or transform.is_spatial():
                    images[i] = transform.apply(image, reverse=reverse)

        return images

    def pad(self, image):
        '''
        Pad data so that it is evenly divisibly by crop window. Also returns
        the amount of padding applied to each dimension so that it can be
        used to restore the original image if needed.
        '''
        shape = image.shape[-3:]
        # Smallest pads needed to evenly divide crop
        amount_to_pad = (self.crop_size - shape % self.crop_size) % self.crop_size
        if self.overlap:
            amount_to_pad += self.crop_size
        # Divide pad into equal parts, rounding when necessary
        pads = [(0, 0)] * image.ndim
        # Pad only spatial dimensions
        pads[-3:] = [(i[0], i[0] + i[1])
                     for i in zip(*divmod(amount_to_pad, 2))]
        return np.pad(image, pads, mode='symmetric'), pads

    def unpad(self, image, idx=None):
        '''
        Crop away padding. Input can either be entire volume, or a cropped sample.
        In the latter case, a sample index must be provided to determine how much
        the sample was padded.
        '''
        # Get lower and upper image coordinates
        if idx is not None:
            assert np.array_equal(image.shape[-3:], self.crop_size)
            image_low, image_high = self.get_crop_coordinates(idx)
        else:
            assert np.array_equal(image.shape[-3:], self.shape)
            image_low, image_high = (0, 0, 0), self.shape

        # Get lower and upper bounds of unpadded image
        bound_low, bound_high = np.array([(p[0], self.shape[dim]-p[1])
                                          for dim, p in enumerate(self.pads[-3:])]).T

        # Bound coordinates
        x1, y1, z1 = np.max([image_low, bound_low], axis=0) - image_low
        x2, y2, z2 = np.min([image_high, bound_high], axis=0) - image_low
        return image[..., x1:x2, y1:y2, z1:z2]


class H5Dataset(AxonDataset):
    '''
    Load dataset from H5 file.
    '''
    def __init__(self, path='./data', val=False, test=False,
                 dataset_name_raw='data', dataset_name_truth='truth',
                 train_dir='train', val_dir='val', test_dir='test',
                 add_channel=False, **kwargs):

        # Load appropriate dataset (train, val, or test)
        if test: path = os.path.join(path, test_dir)
        elif val: path = os.path.join(path, val_dir)
        else: path = os.path.join(path, train_dir)
        image, truth = self._load_data(path, dataset_name_raw, dataset_name_truth)
        testing = val or test

        super().__init__(image,
                         truth,
                         testing=testing,
                         **kwargs)

        # Add empty input channel if needed
        assert self.image.ndim == 4  # C, Z, Y, X
        if add_channel:
            self.image = np.pad(self.image, [(0, 1), (0, 0), (0, 0), (0, 0)])

    def _load_data(self, path, dataset_name_raw, dataset_name_truth):
        '''
        Loads dataset from given path. Currently only single-file datasets are supported.
        '''
        for i, fname in enumerate(os.listdir(path)):
            if i == 0:
                with h5py.File(os.path.join(path, fname), 'r') as f:
                    image = np.array(f[dataset_name_raw], dtype=np.float32)
                    truth = np.array(f[dataset_name_truth], dtype=np.float32)
            else:
                warnings.warn(f"Warning: file {fname} was not loaded")

        return image, truth


class SliceDataset(H5Dataset):
    '''
    Returns images with random permutations of slices for self-supervised learning task.
    '''
    def __init__(self, permutations, axis=0, dataset_name_raw='data',
                 dataset_name_truth=None, **kwargs):

        # Some pretraining data may not be labeled
        if not dataset_name_truth:
            dataset_name_truth = 'data'

        self.permutations = permutations

        assert axis in range(3)
        self.axis = axis

        super().__init__(train_dir='pretrain',
                         dataset_name_raw=dataset_name_raw,
                         dataset_name_truth=dataset_name_truth,
                         **kwargs)

        self.vol_ratio = self.image.size/np.product(self.crop_size)
        self.vol_sum = self.image.sum()

    def __getitem__(self, idx):

        idx, image, *data = super().__getitem__(idx)

        # Randomly select permutation
        label = torch.randint(0, len(self.permutations), (1,)).item()
        permutation = self.permutations[label]

        # Apply permutation along slice axis
        if self.axis == 0:
            image = image[..., permutation, :, :]
        elif self.axis == 1:
            image = image[..., :, permutation, :]
        else:
            image = image[..., :, :, permutation]

        return idx, image, label

    def _get_training_augmentations(self):
        '''
        Return a list of transformations to use at training time.
        '''
        return [AffineTransformation(rotation_step=90, random=True)]


class RegressionDataset(H5Dataset):
    '''
    Include an additional output that can be used to learn regression
    on distance from the ground truth centerline.

    Motivated by Sironi et al. (2014)
    '''

    def __init__(self, **kwargs):
        kwargs.pop('include_centerline')
        super().__init__(include_centerline=True,
                         **kwargs)
        # Compute inverse distance transform of centerline
        distance = distance_transform_edt(1-self.centerline)
        max_distance = np.max((self.truth*distance)[self.aux > 0])
        self.centerline = np.array(
            np.where(distance < max_distance, np.exp(6*(1-distance/max_distance))-1, 0),
            dtype=np.float32
        )


class PathDataset(AxonDataset):
    '''
    Given a predicted segmentation, finds critical points and extracts a
    graph structure that can be used to generate True and False candidate paths.
    Samples are drawn by cropping around these paths, and the output image includes
    two channels, representing raw grayscale imagery and the selected path.
    __getitem__ includes an additional output with a binary label for path classification.

    Based on Mosinska et al. (2019)

    https://arxiv.org/abs/1905.03892
    '''
    def __init__(self, image, segmentation, truth, resolution=(1, 1, 1),
                 delta=16, epsilon=8, soft_labels=False, **kwargs):
        self.segmentation = segmentation
        self.resolution = resolution
        self.delta = delta
        self.epsilon = epsilon
        self.soft_labels = soft_labels
        kwargs.pop('include_centerline')
        super().__init__(image, truth, include_centerline=False, **kwargs)
        # Channel 1 is imagery, channel 2 is path proposal
        assert self.image.ndim == 4 and self.image.shape[0] == 2

    def __getitem__(self, idx):

        (x1, y1, z1), (x2, y2, z2) = self.get_crop_coordinates(idx)

        # Crop data
        image = self.image[..., x1:x2, y1:y2, z1:z2].copy()
        truth = self.truth[..., x1:x2, y1:y2, z1:z2]

        # Draw path in channel 2 of image
        path = self.paths[idx] - (x1, y1, z1)
        image[1][tuple(path.T)] = 1.

        if self.augmentation:
            image, truth = self.transform([image, truth], self.get_transforms(idx))

        # Get classification label
        label = self.labels[idx]

        return idx, image, truth, label

    def _get_sample_indices(self):
        '''
        List of coordinates (e.g. origin of crop window) for each sample.
        '''
        sample_indices = []
        for path in self.paths:
            # Compute bounding box around path
            min_bbox = np.min(path, axis=0)
            max_bbox = np.max(path, axis=0) + 1
            diffs = self.crop_size - (max_bbox - min_bbox)
            upper_bound = self.shape - self.crop_size

            if self.testing:
                # Take a centered crop around bounding box; clip to valid coordinates
                origin = np.clip(min_bbox - diffs//2, 0, upper_bound)
            else:
                # Take a random crop around bounding box
                # Note: if bbox_size > crop_size, path will not fit into a
                # single crop, so instead crop as much of path as possible
                low = np.minimum(min_bbox, upper_bound)
                high = np.clip(min_bbox - diffs, 0, upper_bound)
                origin = [sorted(bound) for bound in zip(low, high)]  # range of possible values

            sample_indices.append(origin)

        return sample_indices

    def _preprocess_data(self):
        # Get nodes and paths along centerline
        self.graph = Graph(self.segmentation,
                           resolution=self.resolution,
                           delta=self.delta,
                           epsilon=self.epsilon)
        self.nodes = self.graph.nodes
        self.paths = self.graph.paths
        # Get labels for path classification
        self.labels = self.get_labels()
        self.sample_keys.extend(['nodes', 'paths', 'labels'])
        super()._preprocess_data()
        # Adjust graph coordinates by padding amount
        if hasattr(self, 'pads'):
            self.nodes += self.pads[-3:, 0]
            self.paths = [path + self.pads[-3:, 0] for path in self.paths]

    def get_crop_coordinates(self, idx):
        '''
        Return lower and upper bound coordinates of sample crop for given index.
        e.g. `return (x1, y1, z1), (x2, y2, z2)`
        '''
        if self.testing:
            # Crop test data using sliding window
            lower = self.sample_indices[idx]
        else:
            # Crop training data randomly
            bounds = self.sample_indices[idx]
            lower = [torch.randint(low, high+1, (1,)).item()
                     for low, high in bounds]

        upper = lower + self.crop_size

        return lower, upper

    def get_labels(self):

        labels = []

        for path in self.paths:

            # Overlap of path with ground truth segmentation
            overlap = self.truth[0][tuple(path.T)].sum()/len(path)
            if not self.soft_labels:
                overlap = overlap > 0.9
            labels.append(overlap)

        # Add extra dimension to play nicely with loss functions
        return np.expand_dims(np.array(labels, dtype=np.float32), 1)


class VolumeStitcher:
    '''
    Stitch together segmentation predictions using a sliding window. Uses information
    from the AxonDataset class to automatically add sample to the correct spatial position
    and reverse whatever test-time augmentation was used for the prediction.
    '''
    def __init__(self, dataset, use_hann=True, channel=0):
        assert dataset.testing
        self.dataset = dataset
        self.image = np.zeros(dataset.shape)
        self.channel = channel
        sample_weight = 1.
        if dataset.augmentation:
            sample_weight /= len(self.dataset.test_time_augmentations)
        if dataset.overlap and use_hann:
            window_func = np.hanning
        else:
            window_func = np.ones
        self.window = self.get_window(window_func, dataset.crop_size)*sample_weight
        self.samples = set()

    @staticmethod
    def get_window(window_func, window_size):
        windows = [window_func(i) for i in window_size]
        w_2d = np.outer(windows[0], windows[1])
        w_3d = np.outer(w_2d, windows[2]).reshape(window_size)
        return w_3d

    def add(self, images, indices):
        # Validate image dimensions
        assert np.array_equal(images.shape[-3:], self.dataset.crop_size)
        # Make sure we aren't adding duplicates
        new_samples = set(indices)
        assert len(indices) == len(new_samples) == len(images)
        assert not new_samples & self.samples
        self.samples.update(new_samples)
        # Add each weighted, reverse-transformed sample to the region it came from
        with ThreadPoolExecutor() as executor:
            for sample, (x1, y1, z1), (x2, y2, z2) in executor.map(self._process_image, images, indices):
                self.image[x1:x2, y1:y2, z1:z2] += sample

    def _process_image(self, image, idx):
        # Collapse channels if necessary so only spatial image remains
        image = image.squeeze()
        if image.ndim == 4:
            image = image[self.channel].squeeze()
        # Reverse any transformations that have been applied to image
        if self.dataset.augmentation:
            image = self.dataset.transform([image], self.dataset.get_transforms(idx), reverse=True)[0]
        # Return image with appropriate weighting and coordinates
        image *= self.window
        (x1, y1, z1), (x2, y2, z2) = self.dataset.get_crop_coordinates(idx)
        return image, (x1, y1, z1), (x2, y2, z2)

    def get_image(self):
        try:
            assert len(self.samples) == self.dataset.length
        except:
            print(len(self.samples), self.dataset.length)
            warnings.warn("Retrieving stitched volume but not all samples have been added")
        return self.dataset.unpad(self.image)
