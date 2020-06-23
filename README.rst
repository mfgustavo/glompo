glompo is a Globally Managed Parallel Optimisation algorithm utilising Bayesian statistics and Gaussian Process Regression

Notes on dependencies: GloMPOScope is an optional feature which can be used to monitor and record GloMPO's management of
the optimization. This uses matplotlib using a backend requiring a Qt binding to use this please ensure pyside2 is
installed in your python environment

pip install pyside2

When running the tests the flag --run-minimize can be used to run a test GloMPO minimisation on a MWE. The --save-outs
flag can be added to save the outputs of this test. This flag will also save the movie by test_scope.py for visual
inspection.