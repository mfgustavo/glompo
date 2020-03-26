

import numpy as np
from glompo.core.expkernel import ExpKernel


class TestExpKernel:
    kernel = ExpKernel(0.1, 5.0)

    def test_call(self):
        y_ker = [self.kernel(0, t) for t in range(50)]
        y_ref = [(5/(t+5)) ** 0.1 for t in range(50)]
        for i in range(50):
            assert np.isclose(y_ker[i], y_ref[i])
