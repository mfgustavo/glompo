

from ..generators.random import RandomGenerator
from ..generators.peterbation import PerterbationGenerator
from ..common.namedtuples import Bound
import numpy as np
import pytest


class TestRandom:

    def test_list_bounds(self):
        np.random.seed(1)
        bounds = [[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54], [1e6, 2e6]]
        gen = RandomGenerator(bounds)

        for i in range(50):
            call = gen.generate()
            assert len(call) == len(bounds)
            for j, x in enumerate(call):
                assert x >= bounds[j][0]
                assert x <= bounds[j][1]

    def test_bound_bounds(self):
        np.random.seed(1)
        bounds = [Bound(198.5, 200), Bound(0, 6), Bound(-0.00001, 0.001), Bound(-64.56, -54.54), Bound(1e6, 2e6)]
        gen = RandomGenerator(bounds)

        for i in range(50):
            call = gen.generate()
            assert len(call) == len(bounds)
            for j, x in enumerate(call):
                assert x >= bounds[j][0]
                assert x <= bounds[j][1]

    def test_tuple_bounds(self):
        np.random.seed(2)
        bounds = ((198.5, 200), (0, 6), (-0.00001, 0.001), (-64.56, -54.54), (1e6, 2e6))
        gen = RandomGenerator(bounds)

        for i in range(50):
            call = gen.generate()
            assert len(call) == len(bounds)
            for j, x in enumerate(call):
                assert x >= bounds[j][0]
                assert x <= bounds[j][1]

    def test_array_bounds(self):
        np.random.seed(3)
        bounds = np.array([[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54], [1e6, 2e6]])
        gen = RandomGenerator(bounds)

        for i in range(50):
            call = gen.generate()
            assert len(call) == len(bounds)
            for j, x in enumerate(call):
                assert x >= bounds[j][0]
                assert x <= bounds[j][1]

    def test_invalid_bounds1(self):
        with pytest.raises(ValueError):
            bounds = np.array([[1098.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -54.54], [1e6, 2e6]])
            RandomGenerator(bounds)

    def test_invalid_bounds2(self):
        with pytest.raises(ValueError):
            bounds = np.array([[198.5, 200], [10, 6], [-0.00001, 0.001], [-64.56, -54.54], [1e6, 2e6]])
            RandomGenerator(bounds)

    def test_invalid_bounds3(self):
        with pytest.raises(ValueError):
            bounds = np.array([[198.5, 200], [0, 6], [0.00001, -0.001], [-64.56, -54.54], [1e6, 2e6]])
            RandomGenerator(bounds)

    def test_invalid_bounds4(self):
        with pytest.raises(ValueError):
            bounds = np.array([[198.5, 200], [0, 6], [-0.00001, 0.001], [-64.56, -64.56], [1e6, 2e6]])
            RandomGenerator(bounds)

    def test_invalid_bounds5(self):
        with pytest.raises(ValueError):
            bounds = np.array([[-np.inf, 200], [0, 6], [-0.00001, 0.001], [-64.56, -64.56], [1e6, 2e6]])
            RandomGenerator(bounds)


class TestPerturbation:

    def test_invalid_bounds1(self):
        with pytest.raises(ValueError):
            x0 = [0, -5, 954.54, 6.23e6, -3.45]
            bounds = [[0, 1], [-100, 100], [950, 1000], [0, 7e6], [-10, 0]]
            scale = [1, 25, 2, 1e1, 5]
            PerterbationGenerator(x0, bounds, scale)

    def test_invalid_bounds2(self):
        with pytest.raises(ValueError):
            x0 = [0.01, -500, 954.54, 6.23e6, -3.45]
            bounds = [[0, 1], [-100, 100], [950, 1000], [0, 7e6], [-10, 0]]
            scale = [1, 25, 2, 1e1, 5]
            PerterbationGenerator(x0, bounds, scale)

    def test_invalid_bounds3(self):
        with pytest.raises(ValueError):
            x0 = [0.01, -5, 9540.54, 6.23e6, -3.45]
            bounds = [[0, 1], [-100, 100], [950, 1000], [0, 7e6], [-10, 0]]
            scale = [1, 25, 2, 1e1, 5]
            PerterbationGenerator(x0, bounds, scale)

    def test_invalid_bounds4(self):
        with pytest.raises(ValueError):
            x0 = [0.01, -5, 954.54, 6.23e6, -3.45]
            bounds = [[0, 1], [-100, 100], [1000, 950], [0, 7e6], [-10, 0]]
            scale = [1, 25, 2, 1e1, 5]
            PerterbationGenerator(x0, bounds, scale)

    def test_invalid_bounds5(self):
        with pytest.raises(ValueError):
            x0 = [0.01, -5, 954.54, 6.23e6, -3.45]
            bounds = [[0, 1], [-100, 100], [950, 1000], [0, np.inf], [-10, 0]]
            scale = [1, 25, 2, 1e1, 5]
            PerterbationGenerator(x0, bounds, scale)

    def test_call(self):
        np.random.seed(6)

        x0 = [0.01, -5, 954.54, 6.23e6, -3.45]
        bounds = [[0, 1], [-100, 100], [950, 1000], [0, 7e6], [-10, 0]]
        scale = [1, 25, 2, 1e6, 5]
        gen = PerterbationGenerator(x0, bounds, scale)

        for i in range(50):
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
        for i in range(5000):
            calls.append(gen.generate())

        assert np.all(np.isclose(np.mean(calls, 0) / x0, 1, rtol=0.1))
