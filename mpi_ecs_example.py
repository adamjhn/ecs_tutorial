""" This example of using NEURON and rxd to model the extracellular space with MPI is based on http://modeldb.yale.edu/238892.

In NEURON 7.7 using MPI creates a copy of the extracellular space on each process and passes the currents to all processes. Increasing the number of processes will reduces the time required to simulate the electrophysiology but not the extracellular diffusion.
"""
from mpi4py import MPI      # MPI must be the first import
from neuron import h, rxd
from neuron.rxd import v
from neuron.rxd.rxdmath import exp, log
from neuron.units import sec

# these additional import are make plots and store the results
from matplotlib import pyplot, colors, colorbar
from matplotlib_scalebar import scalebar
from mpl_toolkits.mplot3d import Axes3D
import numpy
import os
import sys
import pickle


# When using multiple processes get the relevant id and number of hosts 
pc = h.ParallelContext()
pcid = pc.id()
nhost = pc.nhost()

h.load_file('stdrun.hoc')

# set the number of rxd threads
rxd.nthread(2)      

# enable extracellular rxd (only necessary for NEURON prior to 7.7)
rxd.options.enable.extracellular = True 

numpy.random.seed(pcid)            # use a difference seed for each process

# Model parameters
e = 1.60217662e-19
scale = 1e-14/e
h.celsius = 37

# Persistent Na current parameters
gnap = 0.4e-3*scale
thmp = -40
thhp = -48
sigmp = 6
sighp = -6
nap_minf = 1. / (1. + exp(-(v - thmp) / sigmp))
nap_hinf = 1. / (1. + exp(-(v - thhp) / sighp))

# Fast Na current parameters 
gna = 0.3e-3*scale
thm = -34
sigm = 5

# K current parameters 
gk = 5e-3*scale
thn = -55.
sgn = 14.
taun0 = .05
taun1 = .27
thnt = -40
sn = -12
phin = .8
minf = 1. / (1. + exp(-(v - thm) / sigm))
ninf = 1. / (1. + exp(-(v - thn) / sgn))
taun = taun0 + taun1 / (1 + exp(-(v - thnt) / sn))

# K-leak parameters 
gl = 0.1e-3*scale
el = -70

# create a directory to write results
outdir = os.path.relpath('results')
if pcid == 0 and not os.path.exists(outdir):
    try:
        os.makedirs(outdir)
    except:
        print("Unable to create the directory %r for the data and figures"
              % outdir)
        os._exit(1)

# Simulation parameters
Lx, Ly, Lz = 500, 500, 50          # size of the extracellular space mu m^3
Kceil = 15.0                       # threshold used to determine wave speed
Ncell = int(9e4*(Lx*Ly*Lz*1e-9))   # number of neurons (90'000 per mm^3)
Nrec = 1000                        # number of neurons to record/plot
somaR = 25.0                       # soma radius

# Extracellular rxd
# Where? -- define the extracellular space
ecs = rxd.Extracellular(-Lx/2.0, -Ly/2.0,
                        -Lz/2.0, Lx/2.0, Ly/2.0, Lz/2.0, dx=10,
                        volume_fraction=0.2, tortuosity=1.6) 

# Who? -- define the species
k = rxd.Species(ecs, name='k', d=2.62, charge=1, initial=lambda nd: 100 if nd.x3d**2 + nd.y3d**2 + nd.z3d**2 < 100**2 else 3.5)

na = rxd.Species(ecs, name='na', d=1.78, charge=1, initial=134)
ko, nao = k[ecs], na[ecs]

# What?
# No extracellular reactions, just diffusion.

class Neuron:
    """ A neuron with soma and fast and persistent sodium
    currents, potassium currents, passive leak and potassium leak and an
    accumulation mechanism. """
    def __init__(self, x, y, z, rec=False):
        self.x = x
        self.y = y
        self.z = z

        self.soma = h.Section(name='soma', cell=self)
        # add 3D points to locate the neuron in the ECS  
        self.soma.pt3dadd(x, y, z + somaR, 2.0*somaR)
        self.soma.pt3dadd(x, y, z - somaR, 2.0*somaR)
        
        #Where? -- define the intracellular space and membrane
        self.cyt = rxd.Region(self.soma, name='cyt', nrn_region='i')
        self.mem = rxd.Region(self.soma, name='mem', geometry = rxd.membrane())
        cell = [self.cyt, self.mem]
        
        #Who? -- the relevant ions and gates
        self.k = rxd.Species(cell, name='k', d=2.62, charge=1, initial=125)
        self.na = rxd.Species(cell, name='na', d=1.78, charge=1, initial=10)
        self.n = rxd.State(cell, name='n', initial = 0.25512)
        self.ki, self.nai = self.k[self.cyt], self.na[self.cyt]
        
        #What? -- gating variables and ion currents
        self.n_gate = rxd.Rate(self.n, phin * (ninf - self.n) / taun)

        # Nernst potentials
        ena = 1e3*h.R*(h.celsius + 273.15)*log(nao/self.nai)/h.FARADAY
        ek = 1e3*h.R*(h.celsius + 273.15)*log(ko/self.ki)/h.FARADAY

        # Persistent Na current
        self.nap_current = rxd.MultiCompartmentReaction(self.nai, nao, 
                               gnap * nap_minf * nap_hinf * (v - ena),
                               mass_action=False, membrane=self.mem,
                               membrane_flux=True)
        # Na current
        self.na_current  = rxd.MultiCompartmentReaction(self.nai, nao,
                               gna * minf**3 * (1.0 - self.n) * (v - ena),
                               mass_action=False, membrane=self.mem,
                               membrane_flux=True)
        # K current
        self.k_current = rxd.MultiCompartmentReaction(self.ki, ko,
                               gk * self.n**4 * (v - ek), 
                               mass_action=False, membrane=self.mem,
                               membrane_flux=True)
        # K leak
        self.k_leak = rxd.MultiCompartmentReaction(self.ki, ko,
                               gl * (v - ek), 
                               mass_action=False, membrane=self.mem,
                               membrane_flux=True)
        
        if rec: # record membrane potential
            self.somaV = h.Vector()
            self.somaV.record(self.soma(0.5)._ref_v, rec)

# Randomly distribute neurons which we record the membrane potential
# every 50ms
rec_neurons = [Neuron(
    (numpy.random.random()*2.0 - 1.0) * (Lx/2.0), 
    (numpy.random.random()*2.0 - 1.0) * (Ly/2.0), 
    (numpy.random.random()*2.0 - 1.0) * (Lz/2.0), 50)
    for i in range(0, int(Nrec/nhost))]

# Randomly distribute the remaining neurons
all_neurons = [Neuron(
    (numpy.random.random()*2.0 - 1.0) * (Lx/2.0),
    (numpy.random.random()*2.0 - 1.0) * (Ly/2.0),
    (numpy.random.random()*2.0 - 1.0) * (Lz/2.0))
    for i in range(int(Nrec/nhost), int(Ncell/nhost))]


pc.set_maxstep(100) # required before finitialize when using multiple processes
h.finitialize(-70)


def progress_bar(tstop, size=40):
    """ report progress of the simulation """
    prog = h.t/float(tstop)
    fill = int(size*prog)
    empt = size - fill
    progress = '#' * fill + '-' * empt
    sys.stdout.write('[%s] %2.1f%% %6.1fms of %6.1fms\r' % (progress, 100*prog, pc.t(0), tstop))
    sys.stdout.flush()

def plot_rec_neurons():
    """ Produces plots of record neurons membrane potential (shown in figure 1C) """

    # load all the recorded neuron data
    somaV, pos = [], []
    for i in range(nhost):
        fin = open(os.path.join(outdir,'membrane_potential_%i.pkl' % i),'rb')
        [sV, p] = pickle.load(fin)
        fin.close()
        somaV.extend(sV)
        pos.extend(p)

    for idx in range(somaV[0].size()):
        if idx%nhost != pcid:
            continue 
        # create a plot for each record (100ms)
        fig = pyplot.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.set_position([0.0,0.05,0.9,0.9])
        ax.set_xlim([-Lx/2.0, Lx/2.0])
        ax.set_ylim([-Ly/2.0, Ly/2.0])
        ax.set_zlim([-Lz/2.0, Lz/2.0])
        ax.set_xticks([int(Lx*i/4.0) for i in range(-2,3)])
        ax.set_yticks([int(Ly*i/4.0) for i in range(-2,3)])
        ax.set_zticks([int(Lz*i/4.0) for i in range(-2,3)])

        cmap = pyplot.get_cmap('jet')
        for i in range(Nrec):
            x = pos[i]
            scolor = cmap((somaV[i].get(idx)+70.0)/70.0)
            # plot the cell 
            ax.plot([x[0]], [x[1]], [x[2]], '.', linewidth=2, color=scolor)
   
        norm = colors.Normalize(vmin=-70,vmax=0)
        pyplot.title('Neuron membrane potentials; t = %gms' % (idx * 100))

        # add a colorbar 
        ax1 = fig.add_axes([0.88,0.05,0.04,0.9])
        cb1 = colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm,
                                    orientation='vertical')
        cb1.set_label('mV')
            
        # save the plot
        filename = 'neurons_{:05d}.png'.format(idx)
        pyplot.savefig(os.path.join(outdir,filename))
        pyplot.close()

def plot_image_data(data, min_val, max_val, filename, title):
    """Plot a 2d image of the data"""
    sb = scalebar.ScaleBar(1e-6)
    sb.location='lower left'
    pyplot.imshow(data, extent=k[ecs].extent('xy'), vmin=min_val,
                  vmax=max_val, interpolation='nearest', origin='lower')
    pyplot.colorbar()
    sb = scalebar.ScaleBar(1e-6)
    sb.location='lower left'
    ax = pyplot.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.add_artist(sb)
    pyplot.title(title)
    pyplot.xlim(k[ecs].extent('x'))
    pyplot.ylim(k[ecs].extent('y'))
    pyplot.savefig(os.path.join(outdir,filename))
    pyplot.close()

    
h.dt = 1.0 # use a large time step as we are not focusing on spiking behaviour
           # but on slower diffusion


def run(tstop):
    """ Run the simulations saving figures every 10ms """

    while pc.t(0) < tstop:
        if int(pc.t(0)) % 50 == 0:
            # plot extracellular concentrations averaged over depth every 100ms 
            if pcid == 0:
                plot_image_data(k[ecs].states3d.mean(2), 3.5, 40,
                                'k_mean_%05d' % int(pc.t(0)/50),
                                'Potassium concentration; t = %6.0fms'
                                % pc.t(0))

        if pcid == 0: progress_bar(tstop)
        pc.psolve(pc.t(0)+h.dt)  # run the simulation for 1 time step
        
    if pcid == 0:
        progress_bar(tstop)
        print("\nSimulation complete. Plotting membrane potentials")

    # save membrane potentials
    soma, dend, pos = [], [], []
    for n in rec_neurons:
        soma.append(n.somaV)
        pos.append([n.x,n.y,n.z])
    pout = open(os.path.join(outdir,"membrane_potential_%i.pkl" % pcid),'wb')
    pickle.dump([soma,pos],pout)
    pout.close()
    pc.barrier()    # wait for all processes to save

    # plot the membrane potentials
    plot_rec_neurons()

#run the simulation for 10 seconds
run(10 * sec)
