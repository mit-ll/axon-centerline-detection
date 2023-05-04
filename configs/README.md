# Configuration

`axcend` uses YAML files to store all parameters for configuring the dataset, model, and hyperparameters for an experiment. Common parameters can be found in `schema.yaml`, which is used to validate datatypes and prevent bad values from being entered.

There are 3 main subdicts to define:

- `dataset`, which is passed to `get_dataset()` and determines what type of dataset to use
- `model`, which is passed to `get_model()` and determines what type of model is trained
- `experiment`, which contains variables used by `train.py` to determine how the model should be trained, as well as hyperparameters to tune

## Parameters

Some commonly used parameters include:

### Dataset

- `name`: Name of the dataset class to use (must be defined in `axcend/datasets.py`)
- `path`: Relative path to the directory containing data to load
- `dataset_name_raw`: Key to access imagery data
- `dataset_name_truth`: Key to access ground truth data
- `crop_size`: Window size (as list of dimensions) used for cropping input samples
- `augmentation`: Whether or not to augment input samples
- `include_centerline`: Whether or not to include a skeletonized version of the ground truth as an additional input

`PathDataset` only:
- `checkpoint_path`: Stores and loads a precomputed dataset (used to avoid repeated slow initialization)
- `soft_labels`: Uses percent overlap rather than a binary threshold to determine truth value of each path
- `resolution`: Resolution of the imagery in microns
- `add_channel`: Adds a second input channel to be used for path candidates

`SliceDataset` only:
- `axis`: Axis to slice along
- `num_permutations`: Number of permutation classes to create
- `slice_size`: Thickness of each slice (divide `crop_size` by `slice_size` along the chosen dimension to get number of slices, which affects computational complexity)

### Model

- `name`: Name of the model class to use (must be defined in `axcend/model.py`)
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels (i.e. classes)
- `out_channels_aux`: Number of output channels in the auxiliary classifier
- `input_size`: Size of input samples; should be identical to `crop_size` in the dataset config
- `num_levels`: Number of encoder/decoder modules; more levels results in a deeper network with more complexity
- `f_maps`: Number of feature maps to use (doubling with each layer); more feature maps results in more complexity
- `pretrained_path`: Relative path to pretrained model
- `pretrained_modules`: Network modules to copy pretrained weights from (any or all of "encoders", "decoders", and "final_conv")
- `pretrained_frozen`: Which modules to freeze weights for transfer learning (only pretrained modules should be frozen)

`CascadingUNet3D` only:
- `num_levels`: A list of two integers representing depth of upstream and downstream U-Nets
- `residual`: If True, use residual 3D U-Nets; otherwise, standard 3D U-Nets will be used

### Experiment

- `name`: Unique name for the experiment; determines where trained models and experiment logs will be saved
- `trainer`: Name of trainer class (defaults to `Trainer` if not specified)
- `metric`: Metric used for early stopping and determining "best" model
- `mode`: How to optimize metric (i.e. "min" or "max")
- `num_epochs`: Maximum number of epochs to run
- `steps_per_epoch`: Number of training steps defining an epoch (in general, samples are drawn randomly from the training set with random augmentation, so the length/granularity of an epoch is arbitrary; this determines how many weight updates will occur before reporting metrics)
- `patience`: Number of epochs that can pass with no improvement in `metric` before training will terminate
- `microbatch_size`: Number of samples (less than or equal to minibatch size) to process at once; used to limit GPU memory usage (may be possible to increase this number during inference when gradient computation is disabled, for greater parallelization)
- `loss`: Parameters determining which loss function to use

Hyperparameters:
- `minibatch_sizes`: List of minibatch sizes to test
- `learning_rates`: List of learning rates to test
- `weight_decays`: List of weight decay parameters to test
