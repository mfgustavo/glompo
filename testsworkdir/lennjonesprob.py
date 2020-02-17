

from scm.params.core.callbacks import EarlyStopping
from scm.params.core.jobcollection import *
from scm.params.core.dataset import *
from scm.params.optimizers.cma import CMAOptimizer
from scm.params.parameterinterfaces.lennardjones import LennardJonesParams
from scm.plams.mol.atom import Atom
from scm.plams.mol.molecule import Molecule

from optsam.paramswrapper import *
import matplotlib.pyplot as plt
import shutil


def main():
    """ Reparameterises a Lennard-Jones engine to describe the interaction of Argon molecules.
        The reference data is taken from a data and not calculated in PLAMS directly."""

    # Get reference data
    distances = []
    ref_energies = []
    weights = []
    with open('inputs/LennardJonesArgon.txt', mode='r') as file:
        for line in file.readlines()[5:]:
            r, energy, sigma = (float(num) for num in line.split('\t'))
            distances.append(r)
            ref_energies.append(energy)
            weights.append(sigma)
    n_data = len(distances)

    # Setup the jobs
    job_col = JobCollection()
    dat_set = DataSet()
    cma_opt = CMAOptimizer()
    gflsopt = GFLSOptimizer(tmax=60*3,
                            imax=200,
                            save_logger='outputs/gfls_logger',
                            gfls_kwargs={'tr_max': 1})
    len_jon = LennardJonesParams(eps_initial=60, eps_range=(2.2, 220))
    for i in range(n_data):
        entry = JCEntry()
        entry.settings.input.AMS.Task = 'SinglePoint'
        entry.reference_engine = 'ArgonTrainingData'

        mol = Molecule()
        mol.add_atom(Atom(symbol='Ar', coords=(0, 0, 0)))
        mol.add_atom(Atom(symbol='Ar', coords=(0, 0, distances[i])))
        entry.molecule = mol
        job_col.add_entry(f'Ar{distances[i]*220000:05.2f}', entry)
        dat_set.add_entry(f"energy('Ar{distances[i]*220000:05.2f}')", weights[i], ref_energies[i], unit='Hartree')

    shutil.rmtree("outputs/LennardJonesArgonReparm", ignore_errors=True)

    fit_fig, fit_ax = plt.subplots()

    fit_ax.scatter(distances, ref_energies, c='black')
    fit_ax.set_xlabel("r")
    fit_ax.set_ylabel("E")
    for i in range(1, -1, -1):
        name = ['CMA', 'GFLS'][i]
        opt_run = Optimization(f"outputs/LennardJonesArgonReparm/{name}",
                               job_col,
                               dat_set,
                               len_jon,
                               [cma_opt, gflsopt][i],
                               ResidualLossFunction(),
                               callbacks=[EarlyStopping(500)],
                               use_pipe=True)
        results = opt_run.optimize()

        print(f"-------------")
        print(f"{name} OUTCOME")
        print(f"-------------")
        print(results.success)
        print(results.x)
        print(results.fx)
        print(f"-------------")

        distances = np.array(distances)
        eps, r_min = tuple(results.x)
        lj_pred = eps * ((r_min/distances) ** 12 - 2 * (r_min/distances) ** 6)
        fit_ax.plot(distances, lj_pred, label=name)
    fit_ax.legend()
    plt.show()


def plot_traj():
    fig, ax = plt.subplots()
    for name in ['CMA', 'GFLS']:
        data = []
        with open(f"outputs/LennardJonesArgonReparm/{name}/data/history.dat", 'r') as file:
            lines = file.readlines()
            for line in lines[1:]:
                data.append(float(line.split(' ')[1]))
        x = range(len(data))
        ax.plot(x, data, label=name)
    ax.set_title("CMA v GFLS Trajectory")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.legend()
    ax.semilogy()
    plt.show()


if __name__ == '__main__':
    main()
