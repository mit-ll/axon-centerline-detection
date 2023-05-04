import os
import logging
import operator
import pickle
import torch
from ray import tune
from collections import defaultdict
from datasets import PathDataset, VolumeStitcher
from utils.misc import get_model, get_loss, get_dataset, select_permutations

torch.backends.cudnn.enabled = True


class Trainer(tune.Trainable):
    '''
    Generic trainer class; implements setup() which loads model, datasets, and optimizer
    from config, and step() which steps through optimization, updating model parameters
    and logging metrics of interest. Ray Tune uses this class to instantiate and train
    models in each trial for hyperparameter tuning.

    By default, model is checkpointed at every step, which ensures best weights can
    always be recovered, but may result in storage bloat that needs to be manually
    cleaned up by the user.
    '''
    def setup(self, config):
        '''
        Config should contain the following key-value pairs:
        - "dataset": dict containing dataset information
        - "model": dict containing model parameters
        - "experiment": dict containing experiment parameters
        As well as any hyperparameters (e.g. batch size, learning rate)
        '''
        try:
            hyperparams = config['hyperparams']
            model = config['model']
            dataset = config['dataset']
            experiment = config['experiment']
            loss = experiment['loss']
        except KeyError as e:
            logging.error('Config format is invalid')
            raise(e)

        # Device configuration
        self.device = torch.device('cuda')
        torch.manual_seed(0)

        # Create model
        self.model = get_model(**model, device=self.device)

        # Loss and optimizer
        self.criterion = get_loss(**loss, reduction='mean')
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=hyperparams.get('learning_rate', 0.001),
                                          weight_decay=hyperparams.get('weight_decay', 0.0001))

        # Load data
        self.minibatch_size = hyperparams.get('minibatch_size', 8)
        self.microbatch_size = min(experiment.get('microbatch_size', float('inf')),
                                   self.minibatch_size)
        self.train_loader = torch.utils.data.DataLoader(dataset=get_dataset(**dataset),
                                                        batch_size=self.minibatch_size,
                                                        num_workers=min(32, os.cpu_count() + 4),
                                                        drop_last=True,
                                                        pin_memory=True,
                                                        shuffle=True)
        # Optionally set maximum number of steps per epoch for training
        self.train_steps_per_epoch = min(experiment.get('steps_per_epoch', float('inf')),
                                         len(self.train_loader))

        val_dataset = dataset.copy()
        val_dataset['val'] = True
        val_dataset['augmentation'] = False  # trade off validation accuracy for time
        self.val_loader = torch.utils.data.DataLoader(dataset=get_dataset(**val_dataset),
                                                      batch_size=self.microbatch_size,
                                                      pin_memory=True,
                                                      shuffle=False)

    def _evaluate(self, prediction, targets, image=None, loader=None):
        '''
        Return all evaluation metrics to be reported.
        Metrics should be reduced using mean.

        'loss' key is special and will be used to optimize the model.
        '''
        if len(targets) == 1:
            # Automatically unpack for single-target loss functions
            targets = targets[0]

        return {'loss': self.criterion(prediction, targets)}

    def _forward(self, image):
        ''' Perform forward pass on the input image. '''
        return self.model(image)

    def _process_epoch(self, loader, train=True):
        '''
        Iterate through data loader, computing loss function and other metrics
        for each sample and updating model parameters once per training step.

        Returns a dictionary containing metrics to report.
        '''

        if train:
            self.model.train()
            steps_per_epoch = self.train_steps_per_epoch
            prefix = 'training_'
        else:
            self.model.eval()
            steps_per_epoch = len(loader)
            prefix = 'validation_'

        epoch_metrics = {}
        batch_size = loader.batch_size
        microbatch_size = self.microbatch_size

        # Loader yields sample index and an iterable containing the raw image + any ground truth data
        for step, (sample_idx, *data) in enumerate(loader):

            # Break up batch into microbatches if memory-constrained
            for i in range(0, batch_size, microbatch_size):
                # Select microbatch
                image, *targets = [element[i:i + microbatch_size].to(self.device) for element in data]

                # Forward pass
                prediction = self._forward(image)

                # Evaluate loss and other metrics to be reported
                metrics = self._evaluate(prediction, targets, image=image, loader=loader)
                for k, v in metrics.items():
                    assert isinstance(v, torch.Tensor)
                    v /= batch_size / microbatch_size  # Normalize across microbatches
                    k = prefix + k  # Metric name, e.g. "training_loss"
                    epoch_metrics[k] = epoch_metrics.get(k, 0.) + v.item() / steps_per_epoch

                # Backward pass accumulates gradients from each microbatch
                if train:
                    metrics['loss'].backward()

            # Step once gradients are accumulated
            if train:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if step + 1 >= steps_per_epoch:
                    break

        return epoch_metrics

    def step(self):
        ''' Report training and validation metrics for one epoch. '''
        metrics = {}
        metrics.update(self._process_epoch(self.train_loader))
        logging.warning(f"Training loss: {metrics['training_loss']:.4f}")
        with torch.no_grad():
            metrics.update(self._process_epoch(self.val_loader, train=False))
        logging.warning(f"Validation loss: {metrics['validation_loss']:.4f}")
        return metrics

    def save_checkpoint(self, tmp_checkpoint_dir):
        ''' Checkpoint model weights; called once per training step by default. '''
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "weights.ckpt")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        ''' How to reload model if training is interrupted. '''
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "weights.ckpt")
        self.model.load_state_dict(torch.load(checkpoint_path))


class SliceTrainer(Trainer):
    '''
    Trainer class for slice-permuting auxiliary task.
    Implements procedure described in Klinghoffer et al. (2020)

    https://arxiv.org/abs/2004.09629
    '''
    def setup(self, config):
        # Compute random permutations given axis and slice size
        axis = config['dataset']['axis']
        # Slight modification allows user to specify slices thicker than 1 voxel
        slice_size = config['dataset']['slice_size']
        slices = config['dataset']['crop_size'][axis]//slice_size
        assert config['dataset']['crop_size'][axis] % slice_size == 0
        num_permutations = config['dataset']['num_permutations']
        config['dataset']['permutations'] = select_permutations(slices, num_permutations, size=slice_size)
        super().setup(config)
        # Loss function for reconstruction task
        self.mse = torch.nn.MSELoss(reduction='mean')

    def _evaluate(self, prediction, targets, image=None, loader=None):
        '''
        Compute loss + accuracy.
        '''
        # First output is reconstruction task
        reconstruction_loss = self.mse(prediction[0], image)
        # Auxiliary output is permutation classifier
        label = targets[0]
        correct = prediction[1].argmax(1) == label
        loss_weight = loader.dataset.vol_ratio * image.sum()/loader.dataset.vol_sum
        ssl_loss = self.criterion(prediction[1], label) * loss_weight

        return {'ssl_loss': ssl_loss,
                'reconstruction_loss': reconstruction_loss,
                'loss': 0.5*ssl_loss + 0.5*reconstruction_loss,
                'accuracy': correct.sum()/correct.size(0)}


class PathTrainer(Trainer):
    '''
    Trainer for joint segmentation & auxiliary path classificaton network
    described in Mosinka et al. (2019)

    https://arxiv.org/abs/1905.03892

    Note: Model should first be pretrained on segmentation task using base Trainer class.
    '''
    def setup(self, config):
        super().setup(config)
        # Replace AxonDataset loaders with PathDataset loaders
        train_dataset = self._get_path_dataset(self.train_loader, config['dataset'])
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=self.minibatch_size,
                                                        num_workers=min(32, os.cpu_count() + 4),
                                                        drop_last=True,
                                                        pin_memory=True,
                                                        shuffle=True)

        config['dataset']['testing'] = True
        config['dataset']['augmentation'] = False
        val_dataset = self._get_path_dataset(self.val_loader, config['dataset'], val=True)
        self.val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=self.microbatch_size,
                                                      pin_memory=True,
                                                      shuffle=False)
        # Auxiliary classifier loss function
        self.aux_criterion = torch.nn.BCEWithLogitsLoss()

    def _get_path_dataset(self, loader, config, val=False):
        '''
        Create path dataset from predictions on H5 dataset.

        Alternatively, load precomputed dataset from the checkpoint file,
        as instantiating PathDataset involves running a costly graph extraction
        algorithm. Specifying `checkpoint_path` in the config file is recommended
        if reusing one pretrained model across multiple trials.

        Inputs:
          - `loader`: DataLoader for axon dataset to be segmented
          - `config`(dict): Dataset config
        Returns:
          - A PathDataset.
        '''
        # If dataset has been previously checkpointed, use that
        checkpoint = config.get('checkpoint_path')
        if checkpoint:
            filename = 'val.ckpt' if val else 'train.ckpt'
            checkpoint = os.path.join(checkpoint, filename)
            if os.path.exists(checkpoint):
                with open(checkpoint, 'rb') as f:
                    return pickle.load(f)

        # Load axon dataset for segmentation
        dataset = loader.dataset
        dataset.set_context(testing=True,
                            augmentation=True,
                            overlap=True)

        # Predict and stitch segmented image
        stitcher = VolumeStitcher(dataset)
        self.model.eval()
        with torch.no_grad():
            for i, (idx, image, target) in enumerate(loader):
                prediction = torch.sigmoid(self.model(image)[0])
                prediction = prediction.squeeze().cpu().detach().numpy()
                stitcher.add(prediction, idx)
        segmentation = stitcher.get_image()

        # Create path dataset
        image = dataset.unpad(dataset.image)
        truth = dataset.unpad(dataset.truth)
        path_dataset = PathDataset(image=image,
                                   truth=truth,
                                   segmentation=segmentation,
                                   **config)

        # If checkpoint path provided, save dataset to reduce future computation
        if checkpoint:
            with open(checkpoint, 'wb') as f:
                pickle.dump(path_dataset, f)

        return path_dataset

    def _evaluate(self, predictions, targets, image=None, loader=None):
        ''' Compute segmentation loss + path classification loss. '''
        seg_loss = self.criterion(predictions[0], targets[0])
        aux_loss = self.aux_criterion(predictions[1], targets[1])
        return {
            'loss': 0.5*seg_loss + 0.5*aux_loss,
            'segmentation_loss': seg_loss,
            'path_classification_loss': aux_loss
        }

    def _forward(self, image):
        # First pass: no path provided -> segmentation
        pathless_image = image.clone()
        pathless_image[:, 1, ...] = 0
        segmentation, _ = self.model(pathless_image)
        # Second pass: path provided -> path classification
        _, path_classification = self.model(image)
        return segmentation, path_classification


class EarlyStopper(tune.Stopper):
    '''
    Early Stopping class that terminates experiments once a specified
    number of epochs ("patience") has elapsed with no improvement
    in the metric.
    '''
    def __init__(self, metric='validation_loss', mode='min',
                 num_epochs=1000, patience=10, **kwargs):
        self.metric = metric
        self.max_iter = num_epochs
        self.patience = patience
        assert mode in ('min', 'max')
        self._comp = operator.lt if mode == 'min' else operator.gt
        self._iter = defaultdict(lambda: 0)
        self._best = defaultdict(lambda: float('inf'))
        self._iter_since_best = defaultdict(lambda: 0)

    def __call__(self, trial_id, result):
        '''
        Returns true if trial should be stopped because:
          a) The early stopping criterion (# of trials without improvement)
             has been exceeded, or
          b) The maximum number of epochs has been reached
        '''
        self._iter[trial_id] += 1
        self._iter_since_best[trial_id] += 1
        result = result[self.metric]
        if self._comp(result, self._best[trial_id]):
            self._best[trial_id] = result
            self._iter_since_best[trial_id] = 0

        if self._iter_since_best[trial_id] >= self.patience:
            logging.info('Early stop criterion triggered')
            return True

        return self._iter[trial_id] >= self.max_iter

    def stop_all(self):
        return False
