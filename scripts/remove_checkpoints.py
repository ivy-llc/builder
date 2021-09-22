#!/usr/bin/env python3

# global
import os
import ivy
import argparse


def prune_checkpoints_in_dir(chkpts_dir, cutoff, last_only):
    print('pruning checkpoints in {}'.format(chkpts_dir))
    checkpoint_fnames = os.listdir(chkpts_dir)
    if len(checkpoint_fnames) == 0:
        print('No checkpoints found in {}'.format(chkpts_dir))
        return
    checkpoint_fnames.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    if last_only:
        [os.remove(os.path.join(chkpts_dir, cfn)) for cfn in checkpoint_fnames[:-1]]
        return
    for checkpoint_fname in checkpoint_fnames:
        checkpoint_val = int(checkpoint_fname.split('-')[-1].split('.')[0])
        if ivy.exists(cutoff) and checkpoint_val > cutoff:
            os.remove(os.path.join(chkpts_dir, checkpoint_fname))


def prune_checkpoints(base_dir, cutoff, last_only, is_chkpts=False):

    if is_chkpts:
        prune_checkpoints_in_dir(base_dir, cutoff, last_only)
        return

    contents = os.listdir(base_dir)
    contents.sort()

    for item in contents:
        if os.path.isdir(item):
            is_chkpts = False
            if item == 'chkpts':
                is_chkpts = True
            prune_checkpoints(os.path.join(base_dir, item), cutoff, last_only, is_chkpts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bd', '--base_dir', type=str,
                        help='The directory containing the checkpoint files.')
    parser.add_argument('-c', '--cutoff', type=int,
                        help='The cutoff value for the checkpoints, above which existing checkpoints will be removed.')
    parser.add_argument('-lo', '--last_only', action='store_true',
                        help='The cutoff value for the checkpoints, above which existing checkpoints will be removed.')
    parsed_args = parser.parse_args()
    if not parsed_args.cutoff and not parsed_args.last_only:
        raise Exception('Either cutoff value must be specified or last_only mode must be set.')
    base_dir = ivy.default(parsed_args.base_dir, os.getcwd())
    prune_checkpoints(base_dir, parsed_args.cutoff, parsed_args.last_only)


if __name__ == '__main__':
    main()
