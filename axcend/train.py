import os
import yaml
import argparse
import cerberus
import logging
import torch
import ray
from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.schedulers import FIFOScheduler
from trainers import Trainer, SliceTrainer, PathTrainer, EarlyStopper

def main():

    parser = argparse.ArgumentParser(prog='train.py', description='Train 3D U-Net.')
    parser.add_argument('config', type=str,
                        help='YAML containing experiment and model specification')
    parser.add_argument('--schema', type=str, default='configs/schema.yaml',
                        help='YAML containing schema to validate config')
    parser.add_argument('--weights_dir', type=str, default='./weights',
                        help='Directory containing saved models')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory where results and checkpoints will be stored')
    parser.add_argument('-r', '--repeat', type=int, default=1,
                        help='Number of times to run experiment.')
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

    # Any relative path must be replaced by absolute path
    weights_path = os.path.abspath(
        os.path.join(args.weights_dir, config['experiment']['name']))
    config['dataset']['path'] = os.path.abspath(config['dataset']['path'])
    if config['model'].get('pretrained_path'):
        config['model']['pretrained_path'] = os.path.abspath(config['model']['pretrained_path'])
    if config['dataset'].get('checkpoint_path'):
        config['dataset']['checkpoint_path'] = os.path.abspath(config['dataset']['checkpoint_path'])
    logging.info(config)

    # Verify weights directory exists; if not, create it
    if not os.path.isdir(weights_path):
        os.makedirs(weights_path)

    # Train the model
    config['hyperparams'] = {
        "learning_rate": tune.grid_search(config['experiment']['learning_rates']),
        "minibatch_size": tune.grid_search(config['experiment']['minibatch_sizes']),
        "weight_decay": tune.grid_search(config['experiment']['weight_decays']),
    }
    ray.init(_temp_dir=os.environ['TMPDIR'] + '/ray')
    try:
        torch.backends.cudnn.enabled = True
        experiment = config['experiment']
        trainer = globals()[experiment.get('trainer', 'Trainer')]
        for i in range(args.repeat):
            # Run hyperparameter tuning experiment
            # `resources_per_trial` can be adjusted based on node specifications
            # or decreased to allow multiple trials to run in parallel on one node
            result = tune.run(
                trainer,
                resources_per_trial={"cpu": 80, "gpu": 2},
                config=config,
                checkpoint_freq=1,
                search_alg=BasicVariantGenerator(),
                scheduler=FIFOScheduler(),
                name=experiment['name'],
                metric=experiment['metric'],
                mode=experiment['mode'],
                stop=EarlyStopper(**experiment),
                local_dir=args.results_dir
            )

            best_trial = result.get_best_trial(scope="all")
            best_logdir = result.get_best_logdir(scope="all")
            results_df = result.trial_dataframes[best_logdir]
            if experiment['mode'] == 'min':
                best_result = results_df[experiment['metric']].argmin()
            else:
                best_result = results_df[experiment['metric']].argmax()
            last_result = results_df.training_iteration.max()
            print(f"Best trial config: {best_trial.config}")
            print(f"Best result: {results_df[experiment['metric']][best_result]}")
            print(f"Final result: {results_df[experiment['metric']][last_result-1]}")

            # Save best and final weights
            output_path = os.path.join(weights_path, str(i))
            if not os.path.isdir(output_path):
                os.makedirs(output_path)

            best_weights_path = os.path.join(best_logdir, f'checkpoint_{best_result}/weights.ckpt')
            best_weights = torch.load(best_weights_path)
            torch.save(best_weights, os.path.join(output_path, "best.ckpt"))

            last_weights_path = os.path.join(best_logdir, f'checkpoint_{last_result}/weights.ckpt')
            last_weights = torch.load(last_weights_path)
            torch.save(last_weights, os.path.join(output_path, "final.ckpt"))

    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
