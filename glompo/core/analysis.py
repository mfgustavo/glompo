import pickle
from pathlib import Path
from typing import Union

import pandas as pd
from dask.dataframe import DataFrame

from .optimizerlogger import OptimizerLogger

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class OptimizationAnalyser:
    """ Builds upon and analyses optimizer trajectories generated during a GloMPO optimization.
        Designed primarily for interactive use through, for example, Jupyter Notebooks.
        Provides a toolbox of data manipulation and data visualisation methods as well as sensitivity analyses for
        insight into the data.

        Attributes
        ----------
        df: DataFrame
            This class acts as wrapper around a dask.dataframe.DataFrame which provides the bulk of the data analysis
            and can be accessed directly via this attribute. The structure of the DataFrame is:
                pt_id: int (index) - Sequential unique identifier as points are added to the collection.
                set_id: int
                    Identifier for which set the point belongs to (a new set_id is applied for each call of add_set.
                opt_id: int
                    Optimizer identification number within the set.
                iter_id: int
                    Iteration number within the optimizer trajectory.
                eval_id: int
                    Function evaluation number within the set.
                x0...xn: float
                    Function input vector.
                fx: float
    """

    @classmethod
    def load(cls, path: Union[str, Path]):
        """ Load previously saved OptimizationAnalysis pickle. """

    def __init__(self):
        """ Initialises an instance of the analyser. """
        self._set_counter = 0
        self.df = DataFrame.from_pandas(pd.DataFrame(), npartitions=1)

    def add_set(self, source: Union[str, Path, OptimizerLogger]):
        """ A 'set' encompasses all the optimizers and iterations of a single GloMPO optimisation.
            The data can be extracted from various sources and a single OptimizationAnalyser can handle multiple sets at
            once to allow larger quantities of data to be analysed simultaneously.

            Parameters
            ----------
            source: Union[str, Path, OptimizerLogger]
                Either:
                    - A Path to a glompo_optimizer_logs directory containing YAML files for each optimizer.
                    - A Path to a glompo_logger.pkl file containing a saved instance of the optimizations
                      OptimizerLogger instance.
                    - A OptimizerLogger instance containing and optimization history.
        """
        if isinstance(source, str):
            source = Path(source)

        if isinstance(source, OptimizerLogger):
            self._process_opt_logger(source)

        elif isinstance(source, Path):
            if source.name == 'glompo_logger.pkl':
                with source.open('rb') as stream:
                    opt_log = pickle.load(stream)
                    assert isinstance(opt_log, OptimizerLogger)
                    self._process_opt_logger(opt_log)

            elif source.is_dir():
                for file in source.iterdir():
                    pass

    def delete_duplicates(self, atol=1e-10):
        """ Large DataFrames become expensive to handle and may hold redundant information.
            This method removes any adjacent points in which all values (in both input and output space) differ by less
            than atol. This can remove points near the end of an optimisation in which results differ only marginally
            and thus not contribute to the analysis.
        """

    def apply_query(self, query: str):
        """ Filter the database according to the given query. Multiple queries can be applied simultaneously. """

    def save(self):
        """ Saves instance of the OptimizationAnalyser as a pickle file. """

    def _process_opt_logger(self, opt_log: OptimizerLogger):
        """ Handles the addition of an OptimizerLogger instance into the class DataFrame. """
        raise NotImplementedError
