"""Prepare the training data and optionally train the GAN."""
import argparse
from pathlib import Path

from globals import PATH_DATA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=PATH_DATA, help='Protein secondary-structure CSV')
    parser.add_argument('--epochs', type=int, default=0, help='GAN training epochs (default: prepare data only)')
    args = parser.parse_args()
    if not Path(args.data).is_file():
        parser.error(f'Dataset not found: {args.data}. See README.md for the required CSV columns.')
    if args.epochs < 0:
        parser.error('--epochs must be nonnegative')

    from utils.data_utilities import prepare_dataset
    inputs = prepare_dataset(args.data)
    print(f'Prepared {len(inputs)} sequences with shape {inputs.shape[1:]}')
    if args.epochs:
        from gan import GAN
        from globals import BATCH_SIZE
        if len(inputs) < BATCH_SIZE:
            parser.error(f'Training requires at least {BATCH_SIZE} prepared sequences.')
        GAN().train(inputs, epochs=args.epochs)


if __name__ == '__main__':
    main()
