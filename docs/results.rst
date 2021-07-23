*******
Results
*******

GloMPO produces various types of results files which can be configured via the manager;
all or none of the following can be produced. A summary human-readable YAML file is the
most basic record of the optimization. It includes all GloMPO settings, the final result,
computational resources used, checkpoints created, as well as time and date information.

Image files of the optimizer trajectories can also produced, this requires the `matplotlib`
package and is a helpful way to analyze the optimization performance at a glance.

Finally, all iteration and metadata information from the optimizers themselves is now
saved in a compressed HDF5 format. This is more flexible and user-friendly than the
previous YAML files created by v2 GloMPO. This file also contains all the manager metadata;
in this way all information from an optimization can be accessed from one location. To work
with these files within a Python environment, we recommend loading it with the
`Pytables` module. To explore the file in a user-friendly GUI, we recommend using
the `vitables` package.
