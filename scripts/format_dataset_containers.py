#!/usr/bin/env python3

# global
import logging
import os
import ivy
import json
import argparse


def format_containers(cont_dir, cont_format, cont_format_file):

    if cont_format:
        key_chains = ivy.Container(json.loads(cont_format))
    else:
        key_chains = ivy.Container.from_disk_as_json(cont_format_file)

    cont_fnames = os.listdir(cont_dir)
    cont_fnames.sort()
    num_conts = len(cont_fnames)
    num_logs = 100
    log_freq = max((num_conts/num_logs), 1)

    for i, cont_fname in enumerate(cont_fnames):
        if i % log_freq == 0:
            logging.info('reformatting container {} of {}...'.format(i, num_conts))
        cont_fpath = os.path.join(cont_dir, cont_fname)
        cont = ivy.Container.from_disk_as_json(cont_fpath)
        cont = cont.at_key_chains(key_chains)
        cont.to_disk_as_json(cont_fpath)


def main(container_dir=None, cont_format=None, cont_format_fpath=None):
    if (not cont_format and not cont_format_fpath) or (cont_format and cont_format_fpath):
        raise Exception('Exactly one of format or format_file must be specified, but found {} and {}'.format(
            cont_format, cont_format_fpath))
    container_dir = ivy.default(container_dir, os.getcwd())
    format_containers(container_dir, cont_format, cont_format_fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cd', '--container_dir', type=str,
                        help='The directory containing the container json files to format.')
    parser.add_argument('-f', '--format', type=str,
                        help='The key-chain format for all of the containers to use, specified either as a list of'
                             'key chains or a dict, parsed from the input string.')
    parser.add_argument('-ff', '--format_file', type=str,
                        help='The json filepath containing which contains the required key-chain structure'
                             'in the json dict structure')
    parsed_args = parser.parse_args()
    main(parsed_args.container_dir, parsed_args.format, parsed_args.format_file)
