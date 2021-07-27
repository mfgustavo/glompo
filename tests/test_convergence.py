import pytest

from glompo.common.corebase import _CombiCore
from glompo.common.namedtuples import Result
from glompo.convergence.basechecker import BaseChecker, _AndChecker, _OrChecker
from glompo.convergence.fmax import MaxFuncCalls
from glompo.convergence.ftarget import TargetCost
from glompo.convergence.nconv import NOptConverged
from glompo.convergence.nkills import MaxKills
from glompo.convergence.nkillsafterconv import KillsAfterConvergence
from glompo.convergence.omax import MaxOptsStarted
from glompo.convergence.tmax import MaxSeconds


class PlainChecker(BaseChecker):
    def __init__(self):
        super().__init__()

    def __call__(self, manager) -> bool:
        pass


class TrueChecker(BaseChecker):
    def __init__(self):
        super().__init__()
        self.last_result = True

    def __call__(self, manager) -> bool:
        self.last_result = True
        return True


class FalseChecker(BaseChecker):
    def __init__(self):
        super().__init__()
        self.last_result = False

    def __call__(self, manager) -> bool:
        self.last_result = False
        return False


class FancyChecker(BaseChecker):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a
        self.b = b + c

    def __call__(self, manager) -> bool:
        pass


class LazinessChecker(BaseChecker):
    def __call__(self, manager) -> bool:
        raise NotImplementedError


def any_checker():
    return _OrChecker(PlainChecker(), PlainChecker())


def all_checker():
    return _AndChecker(PlainChecker(), PlainChecker())


class TestBase:
    @pytest.mark.parametrize("base1, base2", [(PlainChecker(), PlainChecker()),
                                              (PlainChecker(), any_checker()),
                                              (any_checker(), PlainChecker()),
                                              (PlainChecker(), all_checker()),
                                              (all_checker(), PlainChecker()),
                                              (any_checker(), all_checker())])
    def test_or(self, base1, base2):
        assert (base1 | base2).__class__.__name__ == "_OrChecker"

    @pytest.mark.parametrize("base1, base2", [(PlainChecker(), PlainChecker()),
                                              (PlainChecker(), any_checker()),
                                              (any_checker(), PlainChecker()),
                                              (PlainChecker(), all_checker()),
                                              (all_checker(), PlainChecker()),
                                              (any_checker(), all_checker())])
    def test_and(self, base1, base2):
        assert (base1 & base2).__class__.__name__ == "_AndChecker"

    @pytest.mark.parametrize("checker, output", [(TrueChecker() | LazinessChecker(), True),
                                                 (FalseChecker() & LazinessChecker(), False)])
    def test_laziness(self, checker, output):
        assert checker(None) == output

    @pytest.mark.parametrize("checker, output", [(PlainChecker(), "PlainChecker()"),
                                                 (any_checker(), "[PlainChecker() | \nPlainChecker()]"),
                                                 (all_checker(), "[PlainChecker() & \nPlainChecker()]"),
                                                 (FancyChecker(1, 2, 3), "FancyChecker(a=1, b=5, c)")])
    def test_str(self, checker, output):
        assert str(checker) == output

    @pytest.mark.parametrize("checker", [PlainChecker(),
                                         PlainChecker() & PlainChecker(),
                                         (PlainChecker() | PlainChecker()) & PlainChecker()])
    def test_iter(self, checker):
        assert all([isinstance(base, PlainChecker) for base in iter(checker)])

    @pytest.mark.parametrize("checker, output", [(PlainChecker(), "PlainChecker() = None"),
                                                 (TrueChecker(), "TrueChecker() = True"),
                                                 (any_checker(), "[PlainChecker() = None | \nPlainChecker() = "
                                                                 "None]"),
                                                 (all_checker(), "[PlainChecker() = None & \nPlainChecker() = "
                                                                 "None]"),
                                                 (FancyChecker(1, 2, 3), "FancyChecker(a=1, b=5, c) = None"),
                                                 (FalseChecker() | FalseChecker() & TrueChecker() | TrueChecker() &
                                                  (TrueChecker() | FalseChecker()), "[[FalseChecker() = False | \n"
                                                                                    "[FalseChecker() = False & \n"
                                                                                    "TrueChecker() = True]] | \n"
                                                                                    "[TrueChecker() = True & \n"
                                                                                    "[TrueChecker() = True | \n"
                                                                                    "FalseChecker() = False]]]")])
    def test_conv_str(self, checker, output):
        assert checker.str_with_result() == output

    def test_reset(self):
        checker = TrueChecker() | FalseChecker()
        assert checker.str_with_result() == "[TrueChecker() = True | \nFalseChecker() = False]"
        checker(None)
        assert checker.str_with_result() == "[TrueChecker() = True | \nFalseChecker() = None]"

    def test_combi_init(self):
        with pytest.raises(TypeError):
            _CombiCore(1, 2)

    def test_convergence(self):
        checker = FalseChecker() | FalseChecker() & TrueChecker() | TrueChecker() & (TrueChecker() | FalseChecker())
        assert checker(None) is True


class TestOthers:
    class Manager:
        def __init__(self):
            self.f_counter = 300
            self.conv_counter = 2
            self.o_counter = 10
            self.t_start = 1584521316.09197
            self.hunt_victims = {1, 2, 3, 5, 6, 7}
            self.t_used = 100
            self.result = Result([0, 0], 0, None, None)

    @pytest.mark.parametrize("checker, output", [(MaxSeconds(session_max=60), True),
                                                 (MaxSeconds(session_max=1e318), False),
                                                 (MaxSeconds(overall_max=1e318), False),
                                                 (MaxSeconds(overall_max=100), True),
                                                 (MaxFuncCalls(200), True),
                                                 (MaxFuncCalls(900), False),
                                                 (NOptConverged(2), True),
                                                 (NOptConverged(9), False),
                                                 (MaxKills(6), True),
                                                 (MaxKills(10), False),
                                                 (MaxOptsStarted(10), True),
                                                 (MaxOptsStarted(20), False),
                                                 (TargetCost(0), True),
                                                 (TargetCost(0.9e-6), True),
                                                 (TargetCost(100), True),
                                                 (TargetCost(-100), False),
                                                 (TargetCost(-2e-6), False)])
    def test_conditions(self, checker, output):
        manager = self.Manager()
        assert checker(manager) == output

    def test_killsafterconv(self):
        checker = KillsAfterConvergence(4, 2)
        manager = self.Manager()
        manager.hunt_victims = set()
        manager.conv_counter = 0
        for kills in range(7):
            manager.hunt_victims.add(kills)
            if kills % 3 == 0:
                manager.conv_counter += 1
            assert not checker(manager)
        manager.hunt_victims.add(100)
        assert checker(manager)
        manager.hunt_victims.add(200)
        manager.hunt_victims.add(300)
        manager.hunt_victims.add(400)
        manager.hunt_victims.add(500)
        assert checker(manager)
        manager.conv_counter += 4
        assert checker(manager)
