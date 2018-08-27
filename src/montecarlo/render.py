#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import os
import subprocess as sp
import random
from argparse import ArgumentParser


def config_scene(scene_path, spp):
    """Modifies Tungsten scene file to save albedo and normal."""

    # Load JSON scene file
    if not scene_path.endswith('.json'):
        raise ValueError('Scene file must be in JSON format')
    with open(scene_path, 'r') as fp:
        scene = json.load(fp)

    scene['renderer']['hdr_output_file'] = 'render.exr'

    # Add output buffers
    scene['renderer']['output_buffers'] = []
    for buffer_type in ['albedo', 'normal']:
        buffer_dict = {}
        buffer_dict['type'] = buffer_type
        buffer_dict['hdr_output_file'] = f'{buffer_type}.exr'
        buffer_dict['sample_variance'] = False
        scene['renderer']['output_buffers'].append(buffer_dict)

    # Update SPP count
    scene['renderer']['spp'] = spp

    # Save new scene
    scene_dir = os.path.dirname(os.path.splitext(scene_path)[0])
    new_scene_file = 'scene_buffers.json'
    new_scene_path = os.path.join(scene_dir, new_scene_file)
    with open(new_scene_path, 'w') as fp:
        json.dump(scene, fp, indent=2)

    return new_scene_path


def batch_render(scene_path, params):
    """Renders <scene_path> n times and save to output directory."""

    # Create render directory, if nonexistent
    if not os.path.isdir(params.output_dir):
        os.mkdir(params.output_dir)

    # Batch render
    for i in range(params.nb_renders):
        # Render with Tungsten
        seed = random.randint(0, 100000)
        render_cmd = f'{params.tungsten} -s {seed} -d {params.output_dir} {scene_path}'
        sp.call(render_cmd.split())

        # Move to render directory
        scene_dir = os.path.dirname(os.path.splitext(scene_path)[0])
        img_id = '{0:04d}'.format(i + 1)
        rename_imgs = ['mv {}/{}.exr {}/{}_{}.exr'.format(params.output_dir, t, params.output_dir, img_id, t) for t in ['render', 'albedo', 'normal']]
        for rename in rename_imgs:
            sp.call(rename.split())


def parse_args():
    """Command-line argument parser for generating scenes."""

    # New parser
    parser = ArgumentParser(description='Monte Carlo rendering generator')

    # Rendering parameters
    parser.add_argument('-t', '--tungsten', help='tungsten renderer full path', default='tungsten', type=str)
    parser.add_argument('-d', '--scene-path', help='scene root path', type=str)
    parser.add_argument('-s', '--spp', help='sample per pixel', default=16, type=int)
    parser.add_argument('-n', '--nb-renders', help='number of renders', default=10, type=int)
    parser.add_argument('-o', '--output-dir', help='output directory', default='../../data/renders', type=str)

    return parser.parse_args()

if __name__ == '__main__':
    """Creates scene files."""

    # Parse render parameters and create scene file
    params = parse_args()
    new_scene_path = config_scene(params.scene_path, params.spp)
    batch_render(new_scene_path, params)
