"""
Copyright 2017 Nicholas Moehle

This file is part of CVXPY-CODEGEN.

CVXPY-CODEGEN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY-CODEGEN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY-CODEGEN.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import cvxpy_codegen.tests.utils as tu
import cvxpy as cvx
from cvxpy_codegen import codegen
import numpy as np
from numpy.random import randn

MODULE = 'cvxpy_codegen.tests.examples.test_actuator'




class TestActuatorAllocation(tu.CodegenTestCase):
    class_name = 'TestActuatorAllocation'
    module = MODULE

    def setup_actuator_allocation(self):
        np.random.seed(0)
        n = 10
        m = 5
        self.A = np.random.randn(m,n)
        self.f = np.random.randn(m)
        self.u_last = np.random.randn(n)
        A = cvx.Parameter((m, n), name='A', value=self.A)
        f = cvx.Parameter(m, name='f', value=self.f)
        u_last = cvx.Parameter(n, name='u_last', value=self.u_last)
        u = cvx.Variable(n, name='u')
        constrs = [A*u == f]
        objective = cvx.norm(u - u_last)

        prob = cvx.Problem(cvx.Minimize(objective), constrs)
        self._test_prob(prob)
        self.optval = prob.solve()
        self.u = u.value
        return prob
    

    def test_actuator_allocation(self):
        prob = self.setup_actuator_allocation()
        self._test_prob(prob)
        self._run_codegen_test(prob, '_test_actuator_allocation')

    def _test_actuator_allocation(self):
        self.setup_actuator_allocation()
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve(A=self.A, f=self.f, u_last=self.u_last)
        self.assertAlmostEqual(stats['objective'], self.optval)
        self.assertAlmostEqualMatrices(self.u, vars['u'])

  

if __name__ == '__main__':
    unittest.main()
