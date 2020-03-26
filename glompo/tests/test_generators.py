

import numpy as np
import pytest

from ..generators.random import RandomGenerator
from ..generators.peterbation import PerterbationGenerator
from ..common.namedtuples import Bound


class TestRandom:

    @pytest.mark.parametrize("bounds", [[[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54], [1e6, 2e6]],
                                        [Bound(198.5, 200), Bound(0, 6), Bound(-0.00001, 0.001),
                                         Bound(-64.56, -54.54), Bound(1e6, 2e6)],
                                        ((198.5, 200), (0, 6), (-0.00001, 0.001), (-64.56, -54.54), (1e6, 2e6)),
                                        np.array([[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54], [1e6,
                                                                                                              2e6]]),
                                        ])
    def test_bounds(self, bounds):
        np.random.seed(1)
        gen = RandomGenerator(bounds)

        for _ in range(50):
            call = gen.generate()
            assert len(call) == len(bounds)
            for j, x in enumerate(call):
                assert x >= bounds[j][0]
                assert x <= bounds[j][1]

    @pytest.mark.parametrize("bounds", [np.array([[1098.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54],
                                                  [1e6, 2e6]]),
                                        np.array([[198.5, 200], [10, 6], [-0.00001, 0.001], [-64.56, -54.54],
                                                  [1e6, 2e6]]),
                                        np.array([[198.5, 200], [0, 6], [0.00001, -0.001], [-64.56, -54.54],
                                                  [1e6, 2e6]]),
                                        np.array([[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -64.56],
                                                  [1e6, 2e6]]),
                                        np.array([[-np.inf, 200], [0, 6], [-0.00001, 0.001], [-64.56, -64.56],
                                                  [1e6, 2e6]])])
    def test_invalid_bounds(self, bounds):
        with pytest.raises(ValueError):
            RandomGenerator(bounds)


class TestPerturbation:

    @pytest.mark.parametrize("x0, bounds, scale",
                             [([0, -5, 954.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [950, 1000], [0, 7e6],
                                                                [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -500, 954.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [950, 1000], [0, 7e6],
                                                                     [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -5, 9540.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [950, 1000], [0, 7e6],
                                                                    [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -5, 954.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [1000, 950], [0, 7e6],
                                                                   [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -5, 954.54, 6.23e6, -3.45], [[0, 1], [-100, 100], [950, 1000], [0, np.inf],
                                                                   [-10, 0]], [1, 25, 2, 1e1, 5]),
                              ([0.01, -5, 954.54, 6.23e6], [[0, 1], [-100, 100], [950, 1000], [0, np.inf],
                                                            [-10, 0]], [1, 25, 2, 1e1, 5])
                              ])
    def test_invalid_bounds(self, x0, bounds, scale):
        with pytest.raises(ValueError):
            PerterbationGenerator(x0, bounds, scale)

    def test_call(self):
        np.random.seed(6)

        x0 = [0.01, -5, 954.54, 6.23e6, -3.45]
        bounds = [[0, 1], [-100, 100], [950, 1000], [0, 7e6], [-10, 0]]
        scale = [1, 25, 2, 1e6, 5]
        gen = PerterbationGenerator(x0, bounds, scale)

        for _ in range(50):
            call = gen.generate()
            assert len(call) == len(bounds)
            for j, x in enumerate(call):
                assert x >= bounds[j][0]
                assert x <= bounds[j][1]

    def test_call_mean(self):
        np.random.seed(7687)

        x0 = [0.02, -5, 954.54, 6.23e6, -3.45]
        bounds = [[0, 1], [-100, 100], [950, 1000], [0, 7e6], [-10, 0]]
        scale = [0.01, 5, 10, 1e5, 1]
        gen = PerterbationGenerator(x0, bounds, scale)

        calls = []
        for _ in range(5000):
            calls.append(gen.generate())

        assert np.all(np.isclose(np.mean(calls, 0) / x0, 1, rtol=0.1))
