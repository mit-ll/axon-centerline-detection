import os
import sys
import yaml
import argparse
import cerberus
import logging
import torch
import h5py
from datasets import VolumeStitcher, AxonDataset
from utils.misc import get_dataset, get_model

'''
Use a saved model to perform inference on a new dataset.
'''

def main():

    parser = argparse.ArgumentParser(prog='predict.py', description='Perform inference on a new dataset.')
    parser.add_argument('config', type=str,
                        help='YAML containing experiment and model specification')
    parser.add_argument('output', type=str,
                        help='Path to output dataset.')
    parser.add_argument('--schema', type=str, default='configs/schema.yaml',
                        help='YAML containing schema to validate config')
    parser.add_argument('--weights-dir', type=str, default='./weights',
                        help='Directory containing saved models')
    parser.add_argument('-f', '--filename', type=str, default='0/best.ckpt',
                        help='File containing checkpointed model')
    args = parser.parse_args()

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
    config['dataset']['test'] = False
    config['dataset']['train_dir'] = 'pretrain'
    config['dataset']['dataset_name_truth'] = config['dataset']['dataset_name_raw']
    logging.info(config)

    # Load model
    weights_path = os.path.abspath(
        os.path.join(args.weights_dir, config['experiment']['name'], args.filename))

    device = torch.device('cuda')
    config.pop('pretrained_path', None)
    config.pop('pretrained_modules', None)
    model = get_model(testing=True,
                      device=device,
                      pretrained_path=weights_path,
                      **config['model'])

    # Load dataset
    dataset = get_dataset(**config['dataset'])
    dataset.set_context(testing=True, augmentation=True, overlap=True)

    # Iterate through dataset and stitch predictions
    stitcher = VolumeStitcher(dataset)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=8,
                                            shuffle=False)
    num_batches = len(loader)
    for i, (idx, image, *targets) in enumerate(loader):
        prediction = model(image.to(device))
        prediction = prediction.squeeze().cpu().detach().numpy()
        stitcher.add(prediction, idx)
        sys.stdout.write("\r[{:{}}] {:.1f}%".format("="*int(100*i/(num_batches-1)), 100, (100*i/(num_batches-1))))
        sys.stdout.flush()

    segmentation = stitcher.get_image()
    with h5py.File(args.output, 'w') as f:
        f.create_dataset('/dataset2', chunks=True, data=segmentation)
        f.create_dataset('/dataset1', chunks=True, data=dataset.unpad(dataset.image).squeeze())

    return 0

if __name__ == "__main__":
    main()
