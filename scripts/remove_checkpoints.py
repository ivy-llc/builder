#!/usr/bin/env python3

# global
import os
import ivy
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--checkpoint_dir', type=str,
                        help='The directory containing the checkpoint files.')
    parser.add_argument('-c', '--cutoff', required=True, type=int,
                        help='The cutoff value for the checkpoints, above which existing checkpoints will be removed.')
    parsed_args = parser.parse_args()
    checkpoint_dir = ivy.default(parsed_args.checkpoint_dir, os.getcwd())
    checkpoint_fnames = os.listdir(checkpoint_dir)
    checkpoint_fnames.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    for checkpoint_fname in checkpoint_fnames:
        checkpoint_val = int(checkpoint_fname.split('-')[-1].split('.')[0])
        if checkpoint_val > parsed_args.cutoff:
            os.remove(os.path.join(checkpoint_dir, checkpoint_fname))
