import logging
import os

import pytest
from glompo.common.logging import SplitOptimizerLogs


class TestSplitLogging:

    def run_log(self, directory, propogate, formatter=logging.Formatter()):
        opt_filter = SplitOptimizerLogs(directory, propagate=propogate, formatter=formatter)
        opt_handler = logging.FileHandler(os.path.join(directory, "propogate.txt"), "w")
        opt_handler.addFilter(opt_filter)

        opt_handler.setLevel('DEBUG')

        logging.getLogger("glompo.optimizers").addHandler(opt_handler)
        logging.getLogger("glompo.optimizers").setLevel('DEBUG')

        logging.getLogger("glompo.optimizers.opt1").debug('8452')
        logging.getLogger("glompo.optimizers.opt2").debug('9216')

    def test_split(self, tmp_path):
        self.run_log(tmp_path, False)
        with open(os.path.join(tmp_path, "optimizer_1.log"), 'r') as file:
            key = file.readline()
            assert key == '8452\n'

        with open(os.path.join(tmp_path, "optimizer_2.log"), 'r') as file:
            key = file.readline()
            assert key == '9216\n'

    def test_formatting(self, tmp_path):
        formatter = logging.Formatter("OPT :: %(message)s :: DONE")
        self.run_log(tmp_path, False, formatter)
        with open(os.path.join(tmp_path, "optimizer_1.log"), 'r') as file:
            key = file.readline()
            assert key == "OPT :: 8452 :: DONE\n"

        with open(os.path.join(tmp_path, "optimizer_2.log"), 'r') as file:
            key = file.readline()
            assert key == "OPT :: 9216 :: DONE\n"

    @pytest.mark.parametrize("propogate", [True, False])
    def test_propogate(self, propogate, tmp_path):
        self.run_log(tmp_path, propogate)
        with open(os.path.join(tmp_path, "propogate.txt"), "r") as file:
            lines = file.readlines()

        if propogate:
            assert lines[0] == '8452\n'
            assert lines[1] == '9216\n'
            assert len(lines) == 2
        else:
            assert len(lines) == 0
