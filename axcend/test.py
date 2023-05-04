import os
import yaml
import argparse
import cerberus
import torch
import logging
import h5py
import cc3d
import numpy as np
from datasets import VolumeStitcher
from metrics import dice_score, cldice_score, rhodice_score, rand_f_score
from utils.misc import get_dataset, get_model
from utils.transforms import Skeletonize

def log_metrics(metrics):
    '''
    Log aggregates given a list of metrics measured across trials.
    '''
    assert type(metrics) in (dict, list)

    # If single trial, log the values
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            logging.info(f"{k}: {v:.4f}")

    # Otherwise, log the mean and standard deviation
    else:
        aggregated = {k: [] for k in metrics[0].keys()}
        for result in metrics:
            for k, v in result.items():
                aggregated[k].append(v)

        for k, v in aggregated.items():
            logging.info(f"{k} mean: {np.mean(v):.4f}, stdev: {np.std(v):.4f}")

def test(path, config, save_results=False, output_file='results.h5'):
    '''
    Inputs:
      - `path`(str): Path to checkpointed model
      - `config`(dict): Config containing model and dataset params
      - `save_results`(bool): If True, saves .h5 containing predictions to model directory
      - `output_file`(str): Name of the file to be saved
    Returns:
      - A dictionary mapping metric name to result.
    '''
    # Load model
    device = torch.device('cuda')
    config.pop('pretrained_path', None)
    config.pop('pretrained_modules', None)
    model = get_model(device=device, pretrained_path=path, **config['model'])

    # Special logic is needed to deal with models that have a secondary output
    # for centerline prediction. Might be more elegant to build this into
    # the model class in the future
    is_cl_predictor = config['model']['name'] == 'CascadingUNet3D'

    # Inference on test dataset
    dataset = get_dataset(**config['dataset'])
    batch_size = config['experiment']['microbatch_size']
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=False)
    stitcher = VolumeStitcher(dataset)  # main (segmentation) prediction
    cl_stitcher = VolumeStitcher(dataset)  # auxiliary prediction
    model.eval()
    with torch.no_grad():
        for i, (idx, image, *targets) in enumerate(loader):
            logging.debug(f"Iteration {i}")
            prediction = model(image)
            # Handle multi-output models
            if is_cl_predictor:
                cl_prediction = prediction[1].squeeze().cpu().detach().numpy()
                cl_stitcher.add(cl_prediction, idx)
                prediction = prediction[0].squeeze().cpu().detach().numpy()
            else:
                prediction = prediction.squeeze().cpu().detach().numpy()
            # Stitch predictions into a single probability map
            stitcher.add(prediction, idx)
    # Get stitched predictions
    predictions = stitcher.get_image()

    # Binarized segmentation masks and centerlines
    seg_p = np.uint8(predictions > 0.5)
    cl_p = Skeletonize(threshold=0.5).apply(predictions)
    seg_gt = np.uint8(dataset.unpad(dataset.truth).squeeze())
    cl_gt = np.squeeze(dataset.unpad(dataset.centerline))

    if is_cl_predictor:
        cl_predictions = cl_stitcher.get_image()

    # Connected components for clustering quality metrics
    labels_p = cc3d.connected_components(seg_p, connectivity=6)
    labels_gt = cc3d.connected_components(seg_gt, connectivity=6)

    metrics = {
        'Dice': dice_score(seg_p, seg_gt),
        'clDice': cldice_score(seg_p, seg_gt, cl_p, cl_gt),
        'rhoDice': rhodice_score(cl_p, cl_gt),
        'Adjusted Rand Score': rand_f_score(labels_p, labels_gt)
    }

    # Visualization
    if save_results:
        output_dir = os.path.dirname(path)
        with h5py.File(os.path.join(output_dir, output_file), 'w') as f:
            # RGB image of centerline false negatives, false positives, and true positives
            fn = cl_gt - (cl_gt*seg_p)
            fp = cl_p - (cl_p*seg_gt)
            tp = cl_p * seg_gt
            result = np.transpose(np.stack([fp, tp, fn]), (1, 2, 3, 0))
            # Add alpha channel
            result = np.pad(result, ((0, 0), (0, 0), (0, 0), (0, 1)))
            result[:, :, :, 3] = result.max(axis=3)
            # Save centerline results, as well as other predictions
            f.create_dataset('/result', data=result, chunks=True)
            f.create_dataset('/segmentation', data=predictions, chunks=True)
            if is_cl_predictor:
                f.create_dataset('/cl_detection', data=cl_predictions, chunks=True)

    return metrics

def main():

    parser = argparse.ArgumentParser(prog='test.py', description='Generate and evaluate predictions on test set.')
    parser.add_argument('config', type=str,
                        help='YAML containing experiment and model specification')
    parser.add_argument('--schema', type=str, default='configs/schema.yaml',
                        help='YAML containing schema to validate config')
    parser.add_argument('--weights_dir', type=str, default='./weights',
                        help='Directory containing saved models')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory where results and checkpoints will be stored')
    parser.add_argument('-f', '--filename', type=str, default='',
                        help='File containing checkpointed model (optional)')
    parser.add_argument('-b', '--use-best', action='store_true',
                        help='Use best weights rather than final weights.')
    parser.add_argument('-o', '--output-file', type=str, default='results.h5',
                        help='Name of output file containing results')
    parser.add_argument('-s', '--save-results', action='store_true',
                        help='Output a file containing predictions.')
    parser.add_argument('-t', '--use-test-time-augmentation', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(args.schema, 'r') as f:
        schema = yaml.safe_load(f)

    validator = cerberus.Validator(schema)
    validator.allow_unknown = True
    if not validator.validate(config):
        logging.error('Config format is invalid')
        raise Exception(validator.errors)

    # Testing configuration
    config['dataset']['test'] = True
    config['dataset']['include_centerline'] = True
    config['dataset']['augmentation'] = args.use_test_time_augmentation
    config['dataset']['overlap'] = True
    config['model']['testing'] = True
    logging.info(config)

    # Path to stored models
    weights_path = os.path.abspath(
        os.path.join(args.weights_dir, config['experiment']['name']))

    # Load specific model checkpoint
    if args.filename:
        weights_path = os.path.join(weights_path, args.filename)
        metrics = \
            test(weights_path,
                  config,
                  save_results=args.save_results,
                  output_file=args.output_file)

    # Otherwise, iterate through subdirectories
    else:
        # Within each subdir, look for best or final checkpoint
        reps = next(os.walk(weights_path))[1]
        filename = 'best.ckpt' if args.use_best else 'final.ckpt'
        # Log results
        metrics = []
        for rep in reps:
            logging.debug(f"Starting rep {rep}")
            result = \
                test(os.path.join(weights_path, rep, filename),
                     config,
                     save_results=args.save_results,
                     output_file=args.output_file)
            logging.info(f"Trial number {rep}:")
            log_metrics(result)
            metrics.append(result)

    log_metrics(metrics)

    return 0

if __name__ == "__main__":
    main()
