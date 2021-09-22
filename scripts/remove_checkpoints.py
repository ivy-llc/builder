#!/usr/bin/env python3

# global
import os
import ivy
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--checkpoint_dir', type=str,
                        help='The directory containing the checkpoint files.')
    parser.add_argument('-c', '--cutoff', type=int,
                        help='The cutoff value for the checkpoints, above which existing checkpoints will be removed.')
    parser.add_argument('-lo', '--last_only', action='store_true',
                        help='The cutoff value for the checkpoints, above which existing checkpoints will be removed.')
    parsed_args = parser.parse_args()
    if not parsed_args.cutoff and not parsed_args.last_only:
        raise Exception('Either cutoff value must be specified or last_only mode must be set.')
    checkpoint_dir = ivy.default(parsed_args.checkpoint_dir, os.getcwd())
    assert checkpoint_dir.split('/')[-1] == 'chkpts'
    checkpoint_fnames = os.listdir(checkpoint_dir)
    if len(checkpoint_fnames) == 0:
        print('No checkpoints found in this directory, exiting.')
        return
    checkpoint_fnames.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    if parsed_args.last_only:
        [os.remove(os.path.join(checkpoint_dir, cfn)) for cfn in checkpoint_fnames[:-1]]
        return
    for checkpoint_fname in checkpoint_fnames:
        checkpoint_val = int(checkpoint_fname.split('-')[-1].split('.')[0])
        if ivy.exists(parsed_args.cutoff) and checkpoint_val > parsed_args.cutoff:
            os.remove(os.path.join(checkpoint_dir, checkpoint_fname))


if __name__ == '__main__':
    main()
