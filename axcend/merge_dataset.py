import argparse
import h5py
import numpy as np
from utils.misc import load_image_file

'''
Merge imagery and truth data to single h5, converting to desired datatypes.
'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('input1', type=str,
                        help='Path to raw imagery.')
    parser.add_argument('input2', type=str,
                        help='Path to ground truth data.')
    parser.add_argument('output', type=str,
                        help='Path to output file.')
    parser.add_argument('--key1', type=str, default='data',
                        help='Key for raw imagery')
    parser.add_argument('--dtype1', type=str, default='float32',
                        help='Data type for raw imagery')
    parser.add_argument('--key2', type=str, default='truth',
                        help='Key for ground truth data')
    parser.add_argument('--dtype2', type=str, default='uint8',
                        help='Data type for ground truth')
    args = parser.parse_args()

    assert args.dtype1 in ('uint8', 'uint16', 'float32')
    assert args.dtype2 in ('uint8', 'float32')
    max_pixel_intensity = {'uint8': 2**8-1,
                           'uint16': 2**16-1,
                           'float32': 1.
                           }

    # Load inputs
    imagery = load_image_file(args.input1, key=args.key1)
    truth = load_image_file(args.input2, key=args.key2)

    # Convert to desired formats
    old_max = imagery.max()
    new_max = max_pixel_intensity[args.dtype1]
    imagery = np.array(imagery/old_max*new_max,
                       dtype=np.dtype(args.dtype1))
    truth = np.array(truth/truth.max(),
                     dtype=np.dtype(args.dtype2))

    # Crop truth to size of image
    if not truth.shape == imagery.shape:
        print(f'Cropping truth of size {truth.shape} to {imagery.shape}')
        truth = truth[tuple(map(slice, imagery.shape))]

    # Save to single h5
    with h5py.File(args.output, 'w') as f:
        f.create_dataset(f'/{args.key1}', data=imagery, chunks=True)
        f.create_dataset(f'/{args.key2}', data=truth, chunks=True)

    return 0

if __name__ == "__main__":
    main()

