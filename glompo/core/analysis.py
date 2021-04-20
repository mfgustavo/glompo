import pickle
from pathlib import Path
from typing import Set, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml

from .optimizerlogger import BaseLogger

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
        _df: DataFrame
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
    def load(cls, path: Union[str, Path]) -> 'OptimizationAnalyser':
        """ Load previously saved OptimizationAnalysis pickle. """
        path = Path(path)
        with path.open('rb') as file:
            oa = pickle.load(file)
        return oa

    def __init__(self):
        """ Initialises an instance of the analyser. """
        self._set_counter = 0
        self._n_x_dims = 0

        self._filters = set()

        self._df = dd.from_pandas(pd.DataFrame(), npartitions=1)

    @property
    def n_xdims(self) -> int:
        return self._n_x_dims

    @property
    def n_pts(self) -> int:
        return len(self.filtered_df)

    @property
    def filters(self) -> Set[str]:
        return self._filters

    @property
    def filtered_df(self) -> dd.DataFrame:
        df = self._df
        for query in self.filters:
            df.query(query)
        return df

    def add_set(self, source: Union[str, Path, BaseLogger]):
        """ A 'set' encompasses all the optimizers and iterations of a single GloMPO optimisation.
            The data can be extracted from various sources and a single OptimizationAnalyser can handle multiple sets at
            once to allow larger quantities of data to be analysed simultaneously.

            Parameters
            ----------
            source: Union[str, Path, BaseLogger]
                Either:
                    - A Path to a glompo_optimizer_logs directory containing YAML files for each optimizer.
                    - A Path to a glompo_logger.pkl file containing a saved instance of the optimizations
                      BaseLogger instance.
                    - A BaseLogger instance containing and optimization history.
        """
        if isinstance(source, str):
            source = Path(source)

        if isinstance(source, BaseLogger):
            self._process_opt_logger(source)

        elif isinstance(source, Path):
            if source.name == 'glompo_logger.pkl':
                with source.open('rb') as stream:
                    opt_log = pickle.load(stream)
                    assert isinstance(opt_log, BaseLogger)
                    self._process_opt_logger(opt_log)

            elif source.is_dir():
                self._process_files(source)

        self._set_counter += 1

    def delete_duplicates(self, atol=1e-10):
        """ Large DataFrames become expensive to handle and may hold redundant information.
            This method removes any adjacent points in which all values (in both input and output space) differ by less
            than atol. This can remove points near the end of an optimisation in which results differ only marginally
            and thus not contribute to the analysis.
        """
        raise NotImplementedError

    def apply_filter(self, query: str):
        """ Filter the database according to the given query. Multiple queries can be applied simultaneously. """
        self._filters.add(query)

    def remove_filter(self, index: int):
        """ Removes the previously applied filter at index positions of filters property. """
        self._filters.remove(index)

    def save(self, path: Union[str, Path]):
        """ Saves instance of the OptimizationAnalyser as a pickle file. """
        path = Path(path)
        with path.open('wb') as file:
            pickle.dump(self, file)

    def _process_opt_logger(self, opt_log: BaseLogger):
        """ Handles the addition of an BaseLogger instance into the class DataFrame. """
        raise NotImplementedError

    def _process_files(self, dir_path: Path):
        """ Handles the reading of optimizer log files into the class DataFrame. """

        for file in dir_path.iterdir():
            with file.open('r') as stream:
                data = yaml.load(stream, Loader)

            if len(data['ITERATION_HISTORY']) < 1:
                continue

            opt_id = int(data['DETAILS']['Optimizer ID'])
            n_dims = len(data['ITERATION_HISTORY'][1]['x'])

            opt_df = pd.DataFrame.from_dict(data['ITERATION_HISTORY'], orient='index')
            opt_df['iter_id'] = opt_df.index.tolist()
            opt_df.rename(columns={'f_call_overall': 'eval_id'}, inplace=True)

            del opt_df['f_call_opt']
            del opt_df['i_best']
            del opt_df['fx_best']

            opt_df['set_id'] = np.full(len(opt_df), self._set_counter)
            opt_df['opt_id'] = np.full(len(opt_df), opt_id)

            opt_df = opt_df[['set_id', 'opt_id', 'iter_id', 'eval_id', 'fx', 'x']]

            x_df = pd.DataFrame(opt_df['x'].tolist(), columns=[f'x{i}' for i in range(n_dims)])
            del opt_df['x']
            opt_df[x_df.columns.tolist()] = x_df.to_numpy()

            if len(self._df) == 0:
                self._df = dd.from_pandas(opt_df, chunksize=1000)
            else:
                self._df = dd.concat([self._df, opt_df], axis=0)
                self._df: dd.DataFrame = self._df.assign(pt_id=1)
                self._df['pt_id'] = self._df.pt_id.cumsum() - 1
                self._df = self._df.set_index('pt_id', sorted=True)
