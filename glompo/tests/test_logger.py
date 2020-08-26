import datetime
import shutil

import numpy as np
import pytest
import yaml

from glompo.core.optimizerlogger import OptimizerLogger
from glompo.optimizers.baseoptimizer import BaseOptimizer
from glompo.optimizers.cmawrapper import CMAOptimizer
from glompo.optimizers.gflswrapper import GFLSOptimizer


class TestLogger:
    log = OptimizerLogger()

    opt0 = GFLSOptimizer(0)
    opt1 = CMAOptimizer(1)
    opt2 = BaseOptimizer

    log.add_optimizer(0, type(opt0).__name__, datetime.datetime.now())
    log.add_optimizer(1, type(opt1).__name__, datetime.datetime.now())
    log.add_optimizer(2, type(opt2).__name__, datetime.datetime.now())

    for i in range(1, 30):
        log.put_iteration(0, i, i, i, i, np.exp(i))
    log.put_metadata(0, "Stop Time", datetime.datetime.now())
    log.put_metadata(0, "End Condition", "tmax condition met")

    for i in range(1, 30):
        log.put_iteration(1, i, i, i, [i, i ** 2], np.sin(i))
    log.put_metadata(1, "Stop Time", datetime.datetime.now())
    log.put_metadata(1, "End Condition", "xtol condition met")

    for i in range(1, 30):
        log.put_iteration(2, i, i, i, np.array([i ** 2, i / 2 + 3.14]), np.tan(i))
    log.put_metadata(2, "Stop Time", datetime.datetime.now())
    log.put_metadata(2, "End Condition", "fmax condition met")

    log.put_message(1, "This is a test of the logger message system")

    def test_save(self):
        self.log.save_optimizer("_tmp/test_logger/success", 1)
        self.log.save_optimizer("_tmp/test_logger/all")

        open("_tmp/test_logger/all/0_GFLSOptimizer.yml", "r")
        open("_tmp/test_logger/all/1_CMAOptimizer.yml", "r")
        open("_tmp/test_logger/all/2_ABCMeta.yml", "r")
        open("_tmp/test_logger/success.yml", "r")

    def test_save_summary(self):
        self.log.save_summary("_tmp/test_logger/summary.yml")
        with open("_tmp/test_logger/summary.yml", "r") as file:
            data = yaml.safe_load(file)
        assert len(data) == 3
        assert all([kw in data[0] for kw in ('end_cond', 'f_calls', 'f_best', 'x_best')])

    def test_history0(self):
        hist = self.log.get_history(0)
        assert [*hist][0] == 1
        assert isinstance([*hist.values()][4], dict)
        assert [*hist.values()][3]['fx'] == np.exp(4)
        assert [*hist.values()][7]['x'] == [8]

    def test_history1(self):
        hist = self.log.get_history(1, "fx")
        assert hist[3] == np.sin(4)

    def test_history2(self):
        self.log.put_iteration(1, 30, 30, 30, [30, 30 ** 2], -5)

        hist = self.log.get_history(1, "i_best")
        assert hist[29] == 30

        hist = self.log.get_history(1, "fx_best")
        assert hist[-1] == -5

    def test_history3(self):
        hist = self.log.get_history(1, "x")
        assert hist[4] == [5, 25]

    def test_history4(self):
        with pytest.raises(KeyError):
            self.log.get_history(1, "not a key")

    def test_message(self):
        assert self.log._storage[1].messages[0] == "This is a test of the logger message system"

    def test_len(self):
        assert len(self.log) == 3

    def test_metadata(self):
        assert self.log.get_metadata(0, "End Condition") == "tmax condition met"

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("_tmp/test_logger", ignore_errors=True)
