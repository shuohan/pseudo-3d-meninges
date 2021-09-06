#!/usr/bin/env python 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image')
parser.add_argument('-s', '--surfaces', nargs='+')
parser.add_argument('-d', '--discretized-surfaces', nargs='+')
parser.add_argument('-o', '--output-dir')
parser.add_argument('-V', '--view', default='axial',
                    choices=['axial', 'coronal', 'sagittal'])
parser.add_argument('-c', '--colors', type=int, nargs='+',
                    default=[[255, 0, 0], [0, 255, 0], [255, 255, 0]])
parser.add_argument('-S', '--slice-ind', default=None, type=int)
args = parser.parse_args()


import numpy as np
import re
import nibabel as nib
import trimesh
import PIL
from improc3d import transform_to_axial, transform_to_coronal, quantile_scale
from improc3d import transform_to_sagittal, crop3d
from pathlib import Path


BCK_COLOR = [0, 0, 0, 0]


def read_vtk_to_trimesh(filename):
    with open(filename) as vtk_file:
        read_points = False
        lines = [l.strip() for l in vtk_file.readlines()]
    num_points = int(re.sub(r'POINTS ([0-9]*) .*', r'\1', lines[4]))
    vertices = [[np.float32(n) for n in line.split()]
                for line in lines[5 : 5 + num_points]]
    num_faces = int(re.sub(r'POLYGONS ([0-9]*) .*', r'\1', lines[5 + num_points]))
    num = int(re.sub(r'POLYGONS [0-9]* ([0-9]*)', r'\1', lines[5 + num_points]))
    assert num_faces * 4 == num
    faces = [[int(n) for n in line.split()[1:]]
             for line in lines[6 + num_points : 6 + num_points + num_faces]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def subdivide(mesh, pitch, max_iter=None):
    max_edge = pitch / 2

    if max_iter is None:
        longest_edge = np.linalg.norm(
            mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]],
            axis=1
        ).max()
        max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)

    v, f = trimesh.remesh.subdivide_to_size(
        mesh.vertices,
        mesh.faces,
        max_edge=max_edge,
        max_iter=max_iter
    )

    hit = v / pitch
    hit = np.round(hit).astype(int)
    return hit


def discretize_mesh(mesh, shape):
    min_v = np.min(np.array(mesh.vertices), axis=0)
    vertices = subdivide(mesh, 1)
    # mesh_discretized = mesh.voxelized(pitch=1.0).matrix
    result = np.zeros(shape)
    result[vertices[:, 0],
           vertices[:, 1],
           vertices[:, 2]] = 1
    return result

def discretize_mesh2(mesh, shape):
    vertices = np.round(mesh.vertices).astype(int)
    result = np.zeros(shape)
    result[vertices[:, 0],
           vertices[:, 1],
           vertices[:, 2]] = 1
    return result


for i in range(len(args.colors)):
    args.colors[i] = [
        BCK_COLOR,
        list(args.colors[i]) + [255] # add alpha
    ]

if args.view == 'axial':
    transform = transform_to_axial
elif args.view == 'coronal':
    transform = transform_to_coronal
elif args.view == 'sagittal':
    transform = transform_to_sagittal

Path(args.output_dir).mkdir(exist_ok=True, parents=True)

print('Load image', args.image)
im_obj = nib.load(args.image)
if args.slice_ind is None:
    args.slice_ind = im_obj.shape[2] // 2
im = transform(im_obj.get_fdata(), im_obj.affine, coarse=True)
im = quantile_scale(im, lower_th=0.0, upper_th=255.0).astype(np.uint8)

d_surfaces = list()
if args.discretized_surfaces is None:
    for surf_fn in args.surfaces:
        print('Load surface', surf_fn)
        surf = read_vtk_to_trimesh(surf_fn)
        d_surf = discretize_mesh(surf, im_obj.shape)
        d_surf_obj = nib.Nifti1Image(d_surf, im_obj.affine, im_obj.header)
        d_surf_fn = Path(surf_fn).name
        d_surf_fn = Path(args.output_dir, d_surf_fn).with_suffix('.nii.gz')
        d_surf_obj.to_filename(d_surf_fn)
        d_surfaces.append(d_surf)
else:
    for surf_fn in args.discretized_surfaces:
        print('Load surface', surf_fn)
        d_surf = nib.load(surf_fn).get_fdata()
        d_surfaces.append(d_surf)

colored_surfaces = list()
for color, d in zip(args.colors, d_surfaces):
    d = transform(d, im_obj.affine, coarse=True)
    d = np.array(color, dtype=np.uint8)[d.astype(int)]
    colored_surfaces.append(d)

im_pil = None
print('Compose')
for surf in colored_surfaces:
    im_slice = im[:, :, args.slice_ind]
    surf_slice = surf[:, :, args.slice_ind]
    # im_pil = compose_image_and_labels(im_slice, surf_slice, 1)
    if im_pil is None:
        im_pil = PIL.Image.fromarray(im_slice).convert('RGBA')
    surf_pil = PIL.Image.fromarray(surf_slice).convert('RGBA')
    im_pil = PIL.Image.alpha_composite(im_pil, surf_pil)

im_pil = im_pil.transpose(PIL.Image.TRANSPOSE)
fn = re.sub(r'\.nii(\.gz)*$', '', Path(args.image).name)
im_pil.save(Path(args.output_dir, fn).with_suffix('.png'))
