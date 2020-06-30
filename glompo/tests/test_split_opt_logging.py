

import logging
import os
import shutil

import pytest

from glompo.common.logging import SplitOptimizerLogs


class TestSplitLogging:

    def run_log(self, propogate):
        opt_filter = SplitOptimizerLogs("diverted_logs", propagate=propogate)
        opt_handler = logging.FileHandler("propogate.txt", "w+")
        opt_handler.addFilter(opt_filter)

        opt_handler.setLevel('DEBUG')

        logging.getLogger("glompo.optimizers").addHandler(opt_handler)
        logging.getLogger("glompo.optimizers").setLevel('DEBUG')

        logging.getLogger("glompo.optimizers.opt1").debug('8452')
        logging.getLogger("glompo.optimizers.opt2").debug('9216')

    def test_split(self):
        self.run_log(False)
        with open('diverted_logs/optimizer_1.log', 'r') as file:
            key = int(file.readline())
            assert key == 8452

        with open('diverted_logs/optimizer_2.log', 'r') as file:
            key = int(file.readline())
            assert key == 9216

    @pytest.mark.parametrize("propogate", [True, False])
    def test_propogate(self, propogate):
        self.run_log(propogate)
        with open("propogate.txt", "r") as file:
            lines = file.readlines()
            if propogate:
                assert lines[0] == '8452\n'
                assert lines[1] == '9216\n'
                assert len(lines) == 2
            else:
                assert len(lines) == 0

    def teardown_method(self):
        try:
            os.remove("propogate.txt")
            shutil.rmtree("diverted_logs", ignore_errors=True)
        except FileNotFoundError:
            pass
