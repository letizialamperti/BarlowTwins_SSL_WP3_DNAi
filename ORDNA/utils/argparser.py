import argparse

from configargparse import ArgumentParser
from typing import Union


# Needed to parse booleans from command line properly
def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_args() -> argparse.Namespace:
    parser = ArgumentParser(description='ORDNA', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--arg_log', default=False, type=str2bool, help='if set to True, save arguments to config file')
    parser.add_argument('--seed', default=0, type=int, help='seed for Pytorch Lightning function seed_everything()')


    # Required paths
    parser.add_argument('--samples_dir', required=True, type=str, help='Parent directory of sample .csv files.')
    parser.add_argument('--labels_file', required=True, type=str, help='Path to the labels file for samples.')

    # Data processing parameters
    parser.add_argument('--sequence_length', required=True, type=int, help='The length of the sequences to be processed.')
    parser.add_argument('--num_classes', required=True, type=int, help='Number of classes for classification.')

    # Model type and configuration
    parser.add_argument('--embedder_type', choices=['barlow_twins', 'triplets'], default='barlow_twins',
                        help='Type of embedding methodology to use.')

    # Model and training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation.')
    parser.add_argument('--sample_subset_size', type=int, default=1000, help='Size of the subsets to extract from the samples.')
    parser.add_argument('--token_emb_dim', type=int, default=8, help='Size of the nucleotide token embeddings.')
    parser.add_argument('--sample_repr_dim', type=int, default=256, help='Dimensionality of the sample representation before the fully connected layers.')
    parser.add_argument('--sample_emb_dim', type=int, default=50, help='Dimensionality of the final embedding output by the model.')

    # Loss and optimization
    parser.add_argument('--barlow_twins_lambda', type=float, default=0.005, help='Lambda hyperparameter for the Barlow Twins loss calculation.')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-4, help='Initial learning rate for the optimizer.')

    # Training control
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to train.')
    parser.add_argument('--accelerator', type=str, default='gpu', help='Specify the accelerator device ("gpu" or "cpu").')

    return parser.parse_args()


def write_config_file(args: argparse.Namespace, path: str = 'config.cfg') -> None:
    with open(path, 'w') as f:
        for k in sorted(args.__dict__):
            if args.__dict__[k] != None:
                print(k, '=', args.__dict__[k], file=f)

