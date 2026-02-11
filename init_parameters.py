#!/usr/bin/env python3

import numpy as np
import argparse
import yaml
import os
import trimesh
from scipy.stats import qmc

######################################################
# set-up input arguments
######################################################
parser = argparse.ArgumentParser()
parser.add_argument('--simnum', dest = 'simnum', default = '00001')
parser.add_argument('--scan-file', dest='scan_file', default='scan_params.txt')
parser.add_argument('--scan-index', dest='scan_index', type=int, default=None)
args = parser.parse_args()

######################################################
# read parameters
######################################################
filename = args.simnum + '/parameter/parameters.yaml'
filename_prms = args.simnum + '/parameter/parameters.prms.yaml'
filename_default = args.simnum + '/parameter/parameters-default.yaml'

os.system(f'cp parameters-default.emb.yaml {filename_default}')
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)

def _coerce_value(key, value, parameters_default):
    if key in parameters_default:
        base = parameters_default[key]
        if isinstance(base, bool):
            return value.lower() in ("1", "true", "yes", "on")
        if isinstance(base, int):
            return int(float(value))
        if isinstance(base, float):
            return float(value)
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def _apply_scan_overrides(parameters_default, scan_file, scan_index):
    if scan_index is None:
        scan_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if scan_index <= 0:
        return
    if not os.path.isfile(scan_file):
        raise FileNotFoundError(f"scan file not found: {scan_file}")

    with open(scan_file, "r", encoding="ascii") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]

    if not lines:
        return

    if scan_index > len(lines):
        raise IndexError(f"scan index {scan_index} exceeds {len(lines)} lines in {scan_file}")

    tokens = lines[scan_index - 1].split()
    if len(tokens) % 2 != 0:
        raise ValueError(f"scan line must be key value pairs, got: '{lines[scan_index - 1]}'")

    for key, value in zip(tokens[0::2], tokens[1::2]):
        if key == "L":
            Lval = _coerce_value("Lx", value, parameters_default)
            parameters_default["Lx"] = Lval
            parameters_default["Ly"] = Lval
            parameters_default["Lz"] = Lval
        else:
            parameters_default[key] = _coerce_value(key, value, parameters_default)

_apply_scan_overrides(parameters_default, args.scan_file, args.scan_index)

with open(filename_default, 'w') as f:
    yaml.dump(parameters_default, f)

######################################################
# define variables
######################################################
rho_water = parameters_default["rho_water"]
rho_gas = parameters_default["rho_gas"]
rhow = parameters_default["rhow"]
rhog = parameters_default["rhog"]
energyFactor = parameters_default["energyFactor"]
kbol = parameters_default["kbol"]
t0 = parameters_default["t0"]
Lx = parameters_default["Lx"]
Ly = parameters_default["Ly"]
Lz = parameters_default["Lz"]
visw = parameters_default["visw"] 
visg = parameters_default["visg"]
ul = parameters_default["ul"]

objType = (parameters_default["objFile"])[0:-4]
subDiv = int(parameters_default["subDiv"])
radP = parameters_default["radp"]
rho_shell = parameters_default["rho_shell"]
th_fac = parameters_default["th_fac"]
shell_th = parameters_default["shell_th"]

fscale = parameters_default["fscale"]
Yt = parameters_default["Yt"]
yt_fac = parameters_default["yt_fac"]
Yl = parameters_default["Yl"]
nu = parameters_default["nu"]
rc = parameters_default["rc"]
k_fsi = parameters_default["k_fsi"]
k_bfac = parameters_default["kb_fac"]

ka_tot = parameters_default["ka_tot"]
kv_tot = parameters_default["kv_tot"]
gamma_dpd = parameters_default["gamma_dpd"]
a3 = parameters_default["a3"]
a4 = parameters_default["a4"]
b1 = parameters_default["b1"]
b2 = parameters_default["b2"]

numObjects = int(parameters_default["numObjects"])
numsteps = parameters_default["numsteps"]
numsteps_eq = parameters_default["numsteps_eq"]
stslik = parameters_default["stslik"]
stslik_eq = parameters_default["stslik_eq"]

######################################################
# fundamental scales
######################################################

ue = energyFactor * kbol * t0
um = rho_water * ul**3 / rhow
ut = np.sqrt(um * ul**2 / ue)
kbt = 1 / energyFactor
uvis = um / ul / ut


visw_dpd = visw / uvis
visg_dpd = visg / uvis

mw = rho_water * ul**3 / (rhow * um)
mg = rho_gas * ul**3 / (rhog * um)

######################################################
# object mesh generation
######################################################

if(objType == 'gv'):
    os.system(f'cd gas_vesicle && bash run.sh {args.simnum} && cd ..')
    os.system(f'cp gas_vesicle/gv.off {args.simnum}/mesh/gv.off')
else:
    os.system(f'cd microbubble && python3 sphere_icosphere.py --simnum {args.simnum} -o ../{args.simnum}/mesh/emb.off -s {subDiv} -r {radP} && cd ..')

objFile = args.simnum + '/mesh/' + objType + '.off'
mesh = trimesh.load(objFile)
nverts = len(mesh.vertices)


tot_area = mesh.area
tot_volume = np.abs(mesh.volume)
mvert = rho_shell * shell_th * tot_area * ul**2 / um / nverts

shell_th = th_fac * shell_th

######################################################
# elasticity & FSI
######################################################
Yt = yt_fac * Yt

rho_surf = nverts / tot_area
gamma_fsi = fscale * 2.0 * visw_dpd * (2 * k_fsi + 1) * (2 * k_fsi + 2) * (2 * k_fsi + 3) * (2 * k_fsi + 4) / (3 * np.pi * rc **4 * rhow * rho_surf)
gamma_fsi_gas = fscale * 2.0 * visg_dpd * (2 * k_fsi + 1) * (2 * k_fsi + 2) * (2 * k_fsi + 3) * (2 * k_fsi + 4) / (3 * np.pi * rc **4 * rhog * rho_surf)

# Elastic constants
ka = fscale * Yt * shell_th / (2 * (1 - nu)) / (ue / ul**2)
mu = fscale * Yt * shell_th / (2 * (1 + nu)) / (ue / ul**2)
kb = k_bfac* fscale * 2.0 / np.sqrt(3) * Yl * shell_th**3 / (12 * (1 - nu**2)) / ue

prms_gv = {
    "ka_tot": float(ka_tot),
    "kv_tot": float(kv_tot),
    "gammaC": float(2 * gamma_dpd),
    "kBT": float(kbt),
    "tot_area": float(tot_area),
    "tot_volume": float(tot_volume),
    "kb": float(kb),
    "ka": float(ka),
    "mu": float(mu),
    "a3": float(a3),
    "a4": float(a4),
    "b1": float(b1),
    "b2": float(b2)
}

if(objType == 'gv'):
    zmax = np.argmax(mesh.vertices[:,2])
    zmin = np.argmin(mesh.vertices[:,2])
    prms_gv.update({"tip1": int(zmin), "tip2": int(zmax), "c": 0.0, "muL": float(mu)})

with open(filename_prms, 'w') as f:
    yaml.dump(prms_gv, f)

######################################################
# random positions (Scaling-aware Sobol)
######################################################
# Ensure numObjects is an integer

print(f"Initializing {numObjects} membranes at Packing fraction:\
      {(numObjects * tot_volume) / (Lx * Ly * Lz):.3f}\n", flush=True)

# Use standard sampler to allow non-power-of-two counts
sampler = qmc.Sobol(d=3, scramble=True)
sample = sampler.random(n=numObjects) # Generates exactly numObjects points

sample[:,0] *= Lx
sample[:,1] *= Ly
sample[:,2] *= Lz

rtol = 2.5 * radP
box = np.array([Lx, Ly, Lz])

def periodic_distance(posA, posB, box):
    rel = posA - posB
    rel -= box * np.round(rel / box) # More efficient vectorized-friendly wrap
    return rel

def relax_overlaps(sample, rtol, box):
    max_iter = 500
    for it in range(max_iter):
        print(f"Relaxation iteration {it}...", flush=True)
        moved = False

        # Keep positions inside box
        sample %= box

        # Build periodic cell-list. Cell size chosen so that pairs within rtol
        # are guaranteed to be either in the same cell or one of the 26 neighbors.
        nc = np.floor(box / rtol).astype(int)
        nc[nc == 0] = 1
        cell_size = box / nc

        # Particle cell indices
        idx = np.floor(sample / cell_size).astype(int)
        # safety clamp
        idx = np.minimum(idx, nc - 1)

        # Map cell -> list of particle indices
        cells = {}
        for p in range(numObjects):
            key = (int(idx[p, 0]), int(idx[p, 1]), int(idx[p, 2]))
            cells.setdefault(key, []).append(p)

        # Offsets to check (self + 26 neighbors)
        offs = (-1, 0, 1)

        for i in range(numObjects):
            ci = (int(idx[i, 0]), int(idx[i, 1]), int(idx[i, 2]))
            for dx in offs:
                x = (ci[0] + dx) % int(nc[0])
                for dy in offs:
                    y = (ci[1] + dy) % int(nc[1])
                    for dz in offs:
                        z = (ci[2] + dz) % int(nc[2])
                        cell_list = cells.get((x, y, z), ())
                        for j in cell_list:
                            if j <= i:
                                continue
                            rel = periodic_distance(sample[j], sample[i], box)
                            r = np.linalg.norm(rel)
                            if r < rtol:
                                corr = 0.5 * (1.05 * rtol - r)
                                if r == 0.0:
                                    # avoid division by zero: pick random direction
                                    unit = np.random.normal(size=3)
                                    unit /= np.linalg.norm(unit)
                                else:
                                    unit = rel / r
                                sample[i] -= corr * unit
                                sample[j] += corr * unit
                                sample[i] %= box
                                sample[j] %= box
                                moved = True

        if not moved:
            break

    return sample

sample = relax_overlaps(sample, rtol, box)

pos_q = []
for i in range(numObjects):
    u, v, w = np.random.random(3)
    quatr = [
        np.sqrt(1 - u) * np.sin(2 * np.pi * v),
        np.sqrt(1 - u) * np.cos(2 * np.pi * v),
        np.sqrt(u)     * np.sin(2 * np.pi * w),
        np.sqrt(u)     * np.cos(2 * np.pi * w)
    ]
    pos_q.append([sample[i,0], sample[i,1], sample[i,2], *quatr])

np.savetxt(args.simnum + '/posq.txt', pos_q)

print(f"Initialized {numObjects} membranes with minimal pairwise distance {rtol:.3f} in a box of size [{Lx}, {Ly}, {Lz}].", flush=True)
######################################################
# write final parameters
######################################################


nevery = int(numsteps/stslik)
nevery_eq = int(numsteps_eq/stslik_eq)

parameters = {
    "ut": float(ut), "ue": float(ue), "um": float(um), "uvis": float(uvis),
    "visw_dpd": float(visw_dpd), "visg_dpd": float(visg_dpd), "kbt": float(kbt),
    "nevery": nevery, "nevery_eq": nevery_eq, "nverts": nverts,
    "mw": float(mw), "mg": float(mg), "mvert": float(mvert),
    "tot_area": float(tot_area), "tot_volume": float(tot_volume),
    "gamma_fsi": float(gamma_fsi), "gamma_fsi_gas": float(gamma_fsi_gas),
    "ka": float(ka), "mu": float(mu), "kb": float(kb)
}

with open(filename, 'w') as f:
    yaml.dump(parameters, f)