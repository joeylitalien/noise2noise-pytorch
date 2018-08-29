#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy
import subprocess as sp
import json
import numpy as np
from PIL import Image
from argparse import ArgumentParser


def config_scene(args):
    """Modifies Tungsten scene file to save albedo and normal."""

    # Load JSON scene file
    if not args.scene_path.endswith('.json'):
        raise ValueError('Scene file must be in JSON format')
    with open(args.scene_path, 'r') as fp:
        scene = json.load(fp)
        
    # Duplicate scene for target configuration
    target = copy.deepcopy(scene)

    # Save either in low or high dynamic range
    if args.hdr:
        scene['renderer']['hdr_output_file'] = 'render.exr'
        target['renderer']['hdr_output_file'] = 'target.exr'
    else:
        scene['camera']['tonemap'] = 'reinhard'
        target['camera']['tonemap'] = 'reinhard'
        scene['renderer']['output_file'] = 'render.png'
        target['renderer']['output_file'] = 'target.png'
        del scene['renderer']['hdr_output_file']
        del target['renderer']['hdr_output_file']

    # Add output buffers
    scene['renderer']['output_buffers'] = []
    for buffer_type in ['albedo', 'normal']:
        buffer_dict = {}
        buffer_dict['type'] = buffer_type
        if args.hdr:
            buffer_dict['hdr_output_file'] = f'{buffer_type}.exr'
        else:
            buffer_dict['ldr_output_file'] = f'{buffer_type}.png'
        buffer_dict['sample_variance'] = False
        scene['renderer']['output_buffers'].append(buffer_dict)
        
    # Update resolution, if requested
    if args.resolution:
        res = scene['camera']['resolution']
        if isinstance(res, int):
            w, h = res, res
        else:
            w, h = res[0], res[1]
        ratio_preserved = w / h == args.resolution[0] / args.resolution[1]
        assert ratio_preserved, 'Resizing image with ratio that doesn\'t match reference'
        scene['camera']['resolution'] = list(args.resolution)
        target['camera']['resolution'] = list(args.resolution)

    # Update SPP count
    scene['renderer']['spp'] = args.spp
    target['renderer']['spp'] = args.spp

    # Save buffer scene configuration
    scene_dir = os.path.dirname(os.path.splitext(args.scene_path)[0])
    buffers_file = f'scene_buffers.json'
    buffers_path = os.path.join(scene_dir, buffers_file)
    with open(buffers_path, 'w') as fp:
        json.dump(scene, fp, indent=2)
    
    # Save target scene configuration
    target_file = f'scene_target.json'
    target_path = os.path.join(scene_dir, target_file)
    with open(target_path, 'w') as fp:
        json.dump(target, fp, indent=2)

    return buffers_path, target_path


def batch_render(buffers_path, target_path, args):
    """Renders scene N times and save to output directory."""

    # Create render directory, if nonexistent
    subdirs = ['render', 'albedo', 'normal', 'target']
    subdirs_paths = [os.path.join(args.output_dir, s) for s in subdirs]
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
        [os.mkdir(s) for s in subdirs_paths]
        
    file_ext = '.exr' if args.hdr else '.png'

    # Batch render
    for i in range(args.nb_renders):
        img_id = '{0:04d}'.format(i + 1)
        
        # Render with Tungsten
        seeds = np.random.randint(0, 1e6, size=2)
        render_source_cmd = f'{args.tungsten} -s {seeds[0]} -d {args.output_dir} {buffers_path}'
        target_fname = os.path.join(subdirs_paths[3], f'{img_id}_target{file_ext}')
        render_target_cmd = f'{args.tungsten} -s {seeds[1]} -d {args.output_dir} {target_path}'
        sp.call(render_source_cmd.split())
        sp.call(render_target_cmd.split())

        # Create renaming/moving commands
        mv_imgs = []
        for name, subdir_path in zip(subdirs, subdirs_paths):
            filename = os.path.join(args.output_dir, name + file_ext)
            dest = os.path.join(subdir_path, f'{img_id}_{name}{file_ext}')
            mv_imgs.append(f'mv {filename} {dest}')

        # Call
        for mv in mv_imgs:
            sp.call(mv.split())

    # Move reference images
    scene_root = os.path.dirname(buffers_path)
    if args.hdr:
        if args.resolution:
            print('Warning: Could not resize OpenEXR, do it manually')
        mv_ref_hdr = f'cp {scene_root}/TungstenRender.exr {args.output_dir}/reference.exr'
        sp.call(mv_ref_hdr.split())
    else:
        if args.resolution:
            ref_ldr = Image.open(f'{scene_root}/TungstenRender.png')
            ref_ldr = ref_ldr.resize(tuple(args.resolution), Image.BILINEAR)
            ref_ldr.save(f'{scene_root}/TungstenRender.png')
        
        mv_ref_ldr = f'cp {scene_root}/TungstenRender.png {args.output_dir}/reference.png'
        sp.call(mv_ref_ldr.split())


def parse_args():
    """Command-line argument parser for generating scenes."""

    # New parser
    parser = ArgumentParser(description='Monte Carlo rendering generator')

    # Rendering parameters
    parser.add_argument('-t', '--tungsten', help='tungsten renderer full path', default='tungsten', type=str)
    parser.add_argument('-d', '--scene-path', help='scene root path', type=str)
    parser.add_argument('-r', '--resolution', help='image resolution (w, h)', nargs='+', type=int)
    parser.add_argument('-s', '--spp', help='sample per pixel', default=16, type=int)
    parser.add_argument('-n', '--nb-renders', help='number of renders', default=10, type=int)
    parser.add_argument('--hdr', help='save as hdr images', action='store_true')
    parser.add_argument('-o', '--output-dir', help='output directory', default='../../data/renders', type=str)
    return parser.parse_args()

if __name__ == '__main__':
    """Creates scene files."""

    # Parse render parameters and create scene file
    args = parse_args()
    buffers_path, target_path = config_scene(args)
    batch_render(buffers_path, target_path, args)
