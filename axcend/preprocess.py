import h5py
import argparse
import numpy as np
from scipy.ndimage import median_filter
from utils.misc import load_image_file

'''
Preprocess TIF or HDF5 file before training or inference.
'''

def main():

    parser = argparse.ArgumentParser(prog='preprocess.py', description='Preprocess raw imagery.')
    parser.add_argument('input', type=str,
                        help='Path to input file.')
    parser.add_argument('output', type=str,
                        help='Path to output file.')
    parser.add_argument('--key', type=str, default='dataset1',
                        help='Name of dataset (if HDF5)')
    args = parser.parse_args()

    data = load_image_file(args.input, key=args.key)
    print(f'Loaded dataset of shape {data.shape} from {args.input}')

    # Clip extreme values
    clips = np.percentile(data, [0.01, 99.99])
    print(f'Clipping data at min {clips[0]} and max {clips[1]}...')
    data[data < clips[0]] = clips[0]
    data[data > clips[1]] = clips[1]

    # Apply median filter
    print('Applying median filter...')
    data = median_filter(data, 3)

    # Scale between 0 and 1
    min_value = np.min(data)
    max_value = np.max(data)
    print(f'Scaling between 0 and 1 with min={min_value} and max={max_value}...')
    data = np.array((data - min_value)/(max_value - min_value), dtype=np.float32)

    # Write to HDF5
    print(f'Writing to {args.output}')
    with h5py.File(args.output, 'w') as f:
         f.create_dataset('data', data=data)

    return 0

if __name__ == "__main__":
    main()
