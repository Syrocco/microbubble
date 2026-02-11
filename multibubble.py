#!/usr/bin/env python3

# ============================================
# Imports
# ============================================
import mirheo as mir
import numpy as np
import trimesh
import yaml
import argparse
import os 

######################################################
# Set-up simulation type: equilibration or restart
######################################################
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)   # only one of --equil or --restart
# if --equil: run equilibration stage
# if --restart: continue from checkpoint folder

group.add_argument('--equil', action = 'store_true', default = None)
group.add_argument('--restart', action = 'store_true', default = None)
parser.add_argument('--simnum', dest = 'simnum', default = '00001')   # simulation index
parser.add_argument('--vacuum', action = 'store_true', default = None) # vacuum mode: no solvent

args = parser.parse_args()

dir_name = args.simnum + '/restart'
# logic to ensure restart mode only applies if folder exists and is non-empty
if(args.restart):
    if os.path.isdir(dir_name):
        if not os.listdir(dir_name):
            print("Directory restart is empty. Changing simulation mode from --restart to --equil.")
            args.restart = False
            args.equil = True
        print("Restarting")
    else:
        args.restart = False
        args.equil = True
        print("Given directory doesn't exist. Changing simulation mode from --restart to --equil.")

######################################################
# Load parameters
######################################################

filename_default = args.simnum + '/parameter/parameters-default.yaml'
with open(filename_default, 'rb') as f:
    parameters_default = yaml.load(f, Loader = yaml.CLoader)   # default parameters
    
filename = args.simnum + '/parameter/parameters.yaml'
with open(filename, 'rb') as f:
    parameters = yaml.load(f, Loader = yaml.CLoader)            # simulation parameters

filename_prms = args.simnum + '/parameter/parameters.prms.yaml'
with open(filename_prms, 'rb') as f:
    prms_emb = yaml.load(f, Loader = yaml.CLoader)             # membrane parameter dictionary

# ----------------------------------------------------
# Compute the forcing applied on the membrane ends
# ----------------------------------------------------
def computeForces(vertices, fraction, force):
    # vertices: positions of all membrane vertices
    # fraction: fraction of vertices on which the load is applied
    # force: magnitude applied on top/bottom region

    vertices= np.array(vertices)
    k = int(fraction * 0.5 * len(vertices))   # number of upper/lower vertices

    # indices of largest/smallest z values
    ind_max = np.argpartition(+vertices[:,2], -k)[-k:]
    ind_min = np.argpartition(-vertices[:,2], -k)[-k:]

    vert2 = np.argmin(vertices[ind_max][:,2])
    print('max vertex', vert2)
    
    forces = np.zeros((len(vertices), 3))

    # +z on top region
    forces[ind_max,2] = +force
    # -z on bottom region
    forces[ind_min,2] = -force
    return forces

# object file naming
objType = (parameters_default["objFile"])[0:-4]       # remove extension
objFile = args.simnum + '/mesh/' + objType + '.off'
# number of steps and dt differ between equilibration and production
numsteps = (parameters_default["numsteps"] if args.restart else parameters_default["numsteps_eq"])
dt = (parameters_default["dt"] if args.restart else parameters_default["dt_eq"])

# domain size
Lx = parameters_default["Lx"]
Ly = parameters_default["Ly"]
Lz = parameters_default["Lz"]

nevery = (parameters["nevery"] if args.restart else parameters["nevery_eq"])

# physical parameters
rhow = parameters_default["rhow"]   # density water
rhog = parameters_default["rhog"]   # density gas
alpha = parameters_default["alpha"]
aii = parameters_default["aii"]     # DPD repulsion

gamma_dpd = parameters_default["gamma_dpd"]
gamma_dpd_gas = parameters_default["gamma_dpd_gas"]
gamma_fsi = parameters["gamma_fsi"]
gamma_fsi_gas = parameters["gamma_fsi_gas"]

rc = parameters_default["rc"]
s = parameters_default["s"] 
s_g = parameters_default["s_g"] 

k_fsi = parameters_default["k_fsi"] 

kbt = parameters["kbt"]             # temperature
obmd_flag = parameters_default["obmd_flag"]
mvert = parameters["mvert"]          # mass per membrane vertex
mw = parameters["mw"]                # mass water particles
mg = parameters["mg"]                # mass gas particles
lj_fac = parameters_default["lj_fac"]

# membrane initial pos + quaternion
pos_q = np.reshape(np.loadtxt(args.simnum + '/posq.txt'), (-1, 7))

# domain decomposition: 1×1×1 MPI tasks
ranks = (1, 1, 1)
domain = (Lx, Ly, Lz)

force = parameters_default["force"]

######################################################
# Checkpoint settings
######################################################
checkpoint_step = numsteps - 1

# Initialize Mirheo coordinator
u = mir.Mirheo(ranks, domain, debug_level = 3, log_filename = args.simnum + '/logs/log', checkpoint_folder = args.simnum + "/restart/", checkpoint_every = checkpoint_step)

# load mesh
mesh = trimesh.load_mesh(objFile)

# determine LJ cutoff scaling based on smallest triangle edge
triangle = mesh.vertices[mesh.faces]
edge1 = triangle[:,1] - triangle[:,0]
edges = np.linalg.norm(edge1, axis=1)
lj_fac = 0.8 * np.min(edges)

# define mesh structure for Mirheo
mesh_emb = mir.ParticleVectors.MembraneMesh(vertices = mesh.vertices.tolist(),
                                            stress_free_vertices = mesh.vertices.tolist(),
                                            faces = mesh.faces.tolist())

# define membrane particle vector
emb = mir.ParticleVectors.MembraneVector("emb", mass = mvert, mesh = mesh_emb)

# initial condition (pos + quaternion)
ic_emb = mir.InitialConditions.Membrane(pos_q)

# register membrane PV
u.registerParticleVector(emb, ic_emb)

######################################################
# Create solvent and gas particles unless vacuum mode is activated
######################################################

if(not args.vacuum):
    # water fluid
    water = mir.ParticleVectors.ParticleVector('water', mass = mw)
    ic_water = mir.InitialConditions.Uniform(number_density = rhow)
    u.registerParticleVector(water, ic_water)

    # gas fluid outside the membrane
    sol2 = mir.ParticleVectors.ParticleVector('sol2', mass = mg)
    ic_outer2 = mir.InitialConditions.Uniform(number_density = rhog)
    u.registerParticleVector(sol2, ic_outer2)

    # belonging checker for gas
    inner_checker_1 = mir.BelongingCheckers.Mesh("inner_checker_1")
    u.registerObjectBelongingChecker(inner_checker_1, emb)

    gas = u.applyObjectBelongingChecker(inner_checker_1, sol2,
                                        correct_every = 0,   # immediate correction
                                        inside = "gas",      # inside membrane = gas
                                        outside = "")        # outside = removed

    # belonging checker for water
    inner_checker_2 = mir.BelongingCheckers.Mesh("inner_solvent_checker_2")
    u.registerObjectBelongingChecker(inner_checker_2, emb)
    u.applyObjectBelongingChecker(inner_checker_2, water,
                                  correct_every = 0,
                                  inside = "none",
                                  outside = "")

######################################################
# Interactions
######################################################

afsi = 0.4 * aii

# assign bending and area/volume constraints
prms_emb["bpress"] = parameters_default["bpress"]

# bubble vs GV model
if(objType == 'gv'):
    int_emb = mir.Interactions.MembraneForces("int_emb", "LimUniaxial", "KantorStressFree", **prms_emb, stress_free = True)
else:
    int_emb = mir.Interactions.MembraneForces("int_emb", "Lim", "KantorStressFree", **prms_emb, stress_free = True)

# define pairwise interactions
if(args.vacuum):
    dpd0 = mir.Interactions.Pairwise('dpd0', rc, kind = "DPD", a = 0.0, gamma = 3 * gamma_dpd, kBT = 0.015 * kbt, power = s)    

dpd = mir.Interactions.Pairwise('dpd', rc, kind = "DPD", a = 0*aii, gamma = 0*gamma_dpd, kBT = kbt, power = s)
dpd_wat = mir.Interactions.Pairwise('dpd_wat', rc, kind = "DPD", a = aii, gamma = gamma_dpd, kBT = kbt, power = s)
dpd_gas = mir.Interactions.Pairwise('dpd_gas', rc, kind = "DPD", a = 0 * aii, gamma = gamma_dpd_gas, kBT = kbt, power = s_g)
dpd_fsi = mir.Interactions.Pairwise('dpd_fsi', rc, kind = "DPD", a = afsi, gamma = gamma_fsi, kBT = kbt, power = k_fsi)
dpd_fsi_gas = mir.Interactions.Pairwise('dpd_fsi_gas', rc, kind = "DPD", a = 0*afsi, gamma = gamma_fsi_gas, kBT = kbt, power = k_fsi)

# LJ self-interaction of membrane
#lj = mir.Interactions.Pairwise('lj', rc, kind = "RepulsiveLJ", epsilon = 0.1, sigma = rc / (2**(1/6)), max_force = 10.0, aware_mode = 'Object')
morse = mir.Interactions.Pairwise('morse', rc, kind = "Morse", De = 0.1, r0 = 0.3, beta = 1.5, max_force = 10.0, aware_mode = 'Object')

# short-range strong LJ for vertex–vertex overlap prevention
lj_int = mir.Interactions.Pairwise('lj_int', lj_fac * rc, kind = "RepulsiveLJ", epsilon = 10000.0, sigma = lj_fac * rc / (2**(1/6)), max_force = 100000.0)

######################################################
# Integrators
######################################################
vv = mir.Integrators.VelocityVerlet('vv')   # standard VV integrator
u.registerIntegrator(vv)

# apply integrators
if(args.restart):
    u.setIntegrator(vv, emb)

if(not args.vacuum):
    u.setIntegrator(vv, water)
    u.setIntegrator(vv, gas)

######################################################
# Register interactions to Mirheo
######################################################

u.registerInteraction(int_emb)
if(args.vacuum):
    u.registerInteraction(dpd0)

u.registerInteraction(dpd)
u.registerInteraction(dpd_wat)
u.registerInteraction(dpd_gas)
u.registerInteraction(dpd_fsi)
u.registerInteraction(dpd_fsi_gas)
u.registerInteraction(morse)
u.registerInteraction(lj_int)

# connect interactions to particle vectors
u.setInteraction(int_emb, emb, emb)
u.setInteraction(morse, emb, emb)
u.setInteraction(lj_int, emb, emb)

if(not args.vacuum):
    u.setInteraction(dpd_wat, water, water)
    u.setInteraction(dpd_gas, gas, gas)
    u.setInteraction(dpd, water, gas)
    u.setInteraction(dpd_fsi, emb, water)
    u.setInteraction(dpd_fsi_gas, emb, gas)

if(args.vacuum):
    u.setInteraction(dpd0, emb, emb)

######################################################
# Reflection boundaries: bouncing solvent off membrane
######################################################

if(not args.vacuum):
    bouncer = mir.Bouncers.Mesh("membrane_bounce", 100, 150, "bounce_maxwell", kBT = kbt)
    u.registerBouncer(bouncer)
    u.setBouncer(bouncer, emb, water)
    u.setBouncer(bouncer, emb, gas)

######################################################
# Compute membrane forcing vector
######################################################

fraction = parameters_default["fraction"]
force = parameters_default["tot_force"] / int(0.5 * fraction * len(mesh.vertices))
forces = computeForces(mesh.vertices, fraction, force).tolist()

######################################################
# EQUILIBRATION RUN
######################################################

if args.equil:
    print('equilibration')

    # pin membrane to avoid drifting
    #unr = mir.Plugins.PinObject.Unrestricted
    #omega = [unr, unr, unr]
    #velocity = [0.0, 0.0, 0.0]

    #u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))

    forces = computeForces(mesh.vertices, fraction, force).tolist()
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))

    # dump membrane XYZ trajectory
    #u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))

    # apply membrane external force
    u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraGVForce", emb, forces))

    # run simulation
    u.run(numsteps, dt = dt)
    print('equilibration finished normally')

######################################################
# RESTART / PRODUCTION RUN
######################################################

if args.restart:
    print('production')

    u.restart(f"{args.simnum}/restart/")     # load checkpoint


#    u.registerPlugins(mir.Plugins.createPinObject('pin', emb, nevery, 'force/', velocity, omega))

    forces = computeForces(mesh.vertices, fraction, force).tolist()
    u.registerPlugins(mir.Plugins.createStats('stats', every = nevery))

    # dump membrane trajectory
    #u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump', emb, nevery, f"trj_eq/sim{args.simnum}"))
    #u.registerPlugins(mir.Plugins.createDumpMesh('mesh_dump', emb, nevery, f"trj_mesh/sim{args.simnum}"))

    # dump water trajectory separately
    # u.registerPlugins(mir.Plugins.createDumpXYZ('xyz_dump_water', water, 10*nevery, f"trj_water/sim{args.simnum}"))
    u.registerPlugins(mir.Plugins.createDumpParticlesWithMesh('mesh_dump_memb', emb, nevery, [], f"{args.simnum}/trj_water/sim"))

    u.registerPlugins(mir.Plugins.createMembraneExtraForce("extraGVForce", emb, forces))

    u.run(numsteps, dt = dt)
    print("restart finishing normally")

