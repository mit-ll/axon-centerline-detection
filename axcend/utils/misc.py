import importlib
import numpy as np
import itertools
import torch
import h5py
import zarr
from PIL import Image
from skimage.io import imread
from scipy.spatial.distance import cdist

def get_class_from_name(name, modules):
    '''
    Returns class from the specified module list with the given name.
    '''
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, name, None)
        if clazz is not None:
            return clazz

def get_loss(name, **config):
    '''
    Instantiate loss function from specified name and config.
    '''
    modules = ['losses', 'torch.nn']
    loss_class = get_class_from_name(name, modules)
    return loss_class(**config)

def get_dataset(name, **config):
    '''
    Instantiate dataset from specified name and config.
    '''
    modules = ['datasets']
    dataset_class = get_class_from_name(name, modules)
    return dataset_class(**config)

def get_model(name, device=None, **config):
    '''
    Instantiate model, set device, and load  pre-trained parameters.
    '''
    # Load appropriate class
    modules = ['model']
    model_class = get_class_from_name(name, modules)

    # Set device (default GPU)
    if device is not None:
        assert isinstance(device, torch.device)
    else:
        device = torch.device('cuda')

    # Instantiate model and load any pre-trained parameters
    model = model_class(**config).to(device)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)

    pretrained_path = config.get('pretrained_path')
    modules = config.get('pretrained_modules', [])
    frozen = config.get('pretrained_frozen', [])
    if pretrained_path:
        load_model_parameters(model, pretrained_path, modules=modules, frozen=frozen)

    return model

def load_image_file(path, key=None):
    '''
    Loads image from one of the supported filetypes: TIFF, HDF5, Imaris, Zarr, or JPEG2000
    '''
    valid_exts = set(['tif', 'tiff', 'h5', 'ims', 'zarr', 'jp2', 'jpx'])
    file_ext = path.split('.')[-1]
    if file_ext not in valid_exts:
        raise IOError(f'Expected file to be one of {valid_exts}, received {file_ext} instead')
    elif file_ext in ('tif', 'tiff'):
        data = imread(path)
    elif file_ext in ('jp2', 'jpx'):
        data = np.asarray(Image.open(path))
    elif file_ext == 'zarr':
        assert key is not None
        # TODO: Add logic for non-nested storage
        zarr_storage = zarr.NestedDirectoryStore(path)
        zarr_group = zarr.group(zarr_storage, overwrite=False)
        data = zarr_group[key][:]
    else:
        assert key is not None
        # HDF5 and Imaris are handled the same way
        with h5py.File(path, 'r') as f:
            data = f[key][:]

    return data

def load_model_parameters(model, path, modules=[], frozen=[]):
    '''
    Load pretrained parameters.
    Inputs:
      - `modules`(list): Pre-trained module names (e.g. ['encoder', 'decoder']).
                         If not specified, all weights will be loaded. Otherwise, only
                         specified modules will be loaded, and other modules will be
                         initialized with random weights.
      - `frozen`(list): Module names to freeze for fine-tuning.
    '''
    model_dict = model.state_dict()

    # If modules not specified, load all weights
    if not modules:
        modules = set([module.split('.')[1] for module in model.state_dict()])
    print(f'Loading pre-trained modules {modules} from {path}')

    # Load pretrained weights that exist within given model and specified modules
    pretrained_dict = {k: v for k, v in torch.load(path).items()
                       if k in model_dict and any(m in k for m in modules)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Freeze specified layers so they are not modified by optimizer
    for module in frozen:
        for p in getattr(model.module, module).parameters():
            p.requires_grad = False
    print("Total parameters: " + str(sum(p.numel() for p in model.parameters())))
    print("Trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

def select_permutations(slices, classes, size=1):

    permutations = np.array(list(itertools.permutations(range(slices))))
    selected = np.empty((classes, slices), dtype=np.uint16)
    result = np.empty((classes, slices*size), dtype=np.uint16)

    for i in range(classes):

        if i == 0:
            idx = np.random.randint(len(permutations))
        else:
            idx = dist.argmax()

        selected[i] = permutations[idx]
        result[i] = [j for p in permutations[idx]
                       for j in range(p*size, p*size+size)]

        permutations = np.delete(permutations, idx, axis=0)
        dist = cdist(selected, permutations, metric='hamming').mean(axis=0).flatten()

    return result
