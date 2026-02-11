#!/usr/bin/python
# -*- coding: utf-8 -*-

import trimesh
import numpy as np
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001')
parser.add_argument('-o', dest = 'fname', type = str, default = 'emb.off')
parser.add_argument('-s', dest = 'subDiv', type = int, default = 4)
parser.add_argument('-r', dest = 'radp', type = float, default = 4.0)
args = parser.parse_args()

filename_default = '../' + args.simnum + '/parameter/parameters-default.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

subDiv = args.subDiv
fname = args.fname
#will generate a sphere using subdivision process on an initial icosahedron structure with 20 faces, 30 edges and 12 vertices
#number of faces: Nf = 20*4**subDiv
#number of edges: Ne = 3*Nf/2 = 30*4**subDiv
#number of vertices: Nv = Ne-Nf+2 = 10*4**subDiv+2

radp = args.radp

m = trimesh.creation.icosphere(subdivisions=subDiv, radius=radp)
    
#np.savetxt('sphere_icosphere.txt',m.vertices)

a = m.export(fname, 'off')

#b = m.export('sphere_icosphere.stl','stl')

# Edges are counted twice in trimesh
print(f'trimesh Nv = {len(m.vertices)}, Nf = {len(m.faces)}, Ne = {int(len(m.edges)/2)}')

print(f'Theory Nv = {2+10*4**subDiv}, Nf = {20*4**subDiv}, Ne = {30*4**subDiv}')

#Further reading   https://sinestesia.co/blog/tutorials/python-icospheres/

