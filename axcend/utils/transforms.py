import numpy as np
from abc import abstractmethod, ABCMeta
from scipy.ndimage import rotate
from skimage.morphology import skeletonize
from functools import reduce
import torch


class Transform(metaclass=ABCMeta):

    def __init__(self, random=False, **kwargs):
        # Should handle 3D images with or without sample and channel dimensions
        # i.e. dims = (N, C, Z, Y, X) where N and C dimensions are optional
        self.valid_dims = set([3, 4, 5])
        self.set_params(random=random, **kwargs)
        self.procedure = self.get_procedure()

    def apply(self, image, reverse=False):
        '''
        Apply transform to image. Can be modified in subclass to only
        transform parts of image (e.g. spatial dimensions).
        '''
        return self._apply(image.copy(), reverse=reverse)

    def _apply(self, image, reverse=False):
        ''' Compose and apply transformation functions '''
        assert image.ndim in self.valid_dims
        procedure = reversed(self.procedure) if reverse else self.procedure
        _transform = reduce(self._compose, self.procedure)
        return _transform(image, reverse)

    @staticmethod
    def _compose(f, g):
        return lambda im, rev: g(f(im, rev), rev)

    @abstractmethod
    def set_params(self, random=False, **kwargs):
        ''' Set params for this transform '''

    @abstractmethod
    def get_procedure(self):
        ''' Return list of functions that will applied to image sequentially '''

    @abstractmethod
    def is_spatial():
        ''' True if transformation is purely spatial '''


class GrayscaleAugmentation(Transform):
    ''' Alter brightness and contrast. '''

    def set_params(self, random=False, contrast=0.5, brightness=0.5, luminance=0.5):
        if random:
            rand = torch.rand(3).numpy()
            contrast = rand[0]
            brightness = rand[1]
            luminance = rand[2]
        assert contrast > 0 and contrast <= 1
        assert brightness > 0 and brightness <= 1
        assert luminance > 0 and luminance <= 1
        self.alpha = 1 + (contrast - 0.5) * 0.3
        self.beta = (brightness / 2) * 0.3  # only positive adjustments
        self.gamma = 2.0**(luminance * 2 - 1)

    def apply(self, image, reverse=False):
        ''' Apply each function to grayscale channel '''
        # If there are multiple channels, only treat first as grayscale
        image = image.copy()
        if image.ndim > 3:
            image[..., 0, :, :, :] = self._apply(image[..., 0, :, :, :], reverse=reverse)
        else:
            image = self._apply(image, reverse=reverse)

        return image

    def get_procedure(self):
        return [self.adjust_contrast, self.adjust_brightness, self.adjust_luminance]

    def adjust_contrast(self, image, reverse=False):
        if reverse:
            return image / self.alpha
        return image * self.alpha

    def adjust_brightness(self, image, reverse=False):
        ''' Note: due to clipping, this function is not fully reversible '''
        if reverse:
            return np.clip(image - self.beta, 0, 1)
        return np.clip(image + self.beta, 0, 1)

    def adjust_luminance(self, image, reverse=False):
        if reverse:
            return image**(1/self.gamma)
        return image**self.gamma

    @staticmethod
    def is_spatial():
        return False


class AffineTransformation(Transform):
    ''' Rotation and flipping. '''

    def set_params(self, random=False, angle=0, flip=[0, 0, 0],
                   rotation_step=1, interpolation=0):
        self.interpolation = interpolation
        if random:
            self.flip = torch.rand(3).numpy() > 0.5
            self.angle = torch.randint(0, 90//rotation_step + 1, (1,)).item()*rotation_step
        else:
            self.flip = flip
            self.angle = angle

    def get_procedure(self):
        return [self.apply_flip, self.apply_rotate]

    def apply_flip(self, image, reverse=False):
        first_spatial_dim = image.ndim - 3
        # Random flipping about each axis
        for axis, flip in enumerate(self.flip):
            image = np.flip(image, axis=axis + first_spatial_dim)
        return image

    def apply_rotate(self, image, reverse=False):
        '''
        Note: rotations are not reversible if angle is not a multiple of
        90 degrees or the dimensions along rotated axes are not equal
        '''
        first_spatial_dim = image.ndim - 3
        # Random X-Y rotation with reflective padding
        axes = (first_spatial_dim + 2, first_spatial_dim + 1)
        angle = self.angle
        if reverse:
            angle *= -1
        return rotate(image, angle, axes=axes, order=self.interpolation,
                      reshape=False, mode='reflect')

    @staticmethod
    def is_spatial():
        return True


class Skeletonize(Transform):
    ''' Morphological thinning '''

    def set_params(self, random=False, threshold=0.5):
        ''' Set params for this transform '''
        self.threshold = threshold

    def apply(self, image, reverse=False):
        '''
        skimage.morphology.skeletonize() requires input to be 3D,
        so if there are multiple samples, we must skeletonize iteratively
        '''
        image = image.copy()
        if image.ndim == 5:
            for i in range(image.shape[0]):
                image[i, 0, :, :, :] = self._apply(image[i, 0, :, :, :], reverse=reverse)
        elif image.ndim == 4:
            image[0, :, :, :] = self._apply(image[0, :, :, :], reverse=reverse)
        else:
            image = self._apply(image, reverse=reverse)

        return image

    def get_procedure(self):
        return [self.binarize, self.thin]

    def binarize(self, image, reverse=False):
        if reverse:
            return image
        return np.array(image > self.threshold, dtype=np.uint8)

    def thin(self, image, reverse=False):
        '''
        Currently not reversible, though could potentially implement
        reverse given a radius parameter
        '''
        if reverse:
            return image
        return skeletonize(image)

    @staticmethod
    def is_spatial():
        return False
