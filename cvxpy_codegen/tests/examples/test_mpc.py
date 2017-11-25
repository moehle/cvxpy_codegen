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

MODULE = 'cvxpy_codegen.tests.examples.test_mpc'


class TestMpc(tu.CodegenTestCase):
    class_name = 'TestMpc'
    module = MODULE

    def mpc_setup(self, objective='1norm'):
        np.random.seed(0)
        n = 3
        m = 2
        T = 6
        self.A = np.eye(n) + .2*np.random.randn(n,n)
        self.B = 5*np.random.randn(n,m)
        self.x0 = 5*np.random.randn(n)

        A = cvx.Parameter((n, n), name='A', value=self.A)
        B = cvx.Parameter((n, m), name='B', value=self.B)
        x0 = cvx.Parameter(n, name='x0', value=self.x0)

        x = cvx.Variable((n, T+1), name='x')
        u = cvx.Variable((m, T), name='u')

        obj = 0
        constr = []
        constr += [x[:,0] == x0]
        for t in range(T):
            constr += [x[:,t+1] == A*x[:,t] + B*u[:,t]]
            constr += [cvx.norm(u[:,t], 'inf') <= 1] 
            if objective == 'quad':
                obj += cvx.sum_squares(x[:,t+1]) + cvx.sum_squares(u[:,t])
            elif objective == '1norm':
                obj += cvx.norm(x[:,t+1], 1) + cvx.norm(u[:,t], 1)
            elif objective == 'exp1norm':
                obj += cvx.exp(cvx.norm(x[:,t+1], 1)) + cvx.norm(u[:,t], 1)
            else:
                raise Exception

        prob = cvx.Problem(cvx.Minimize(obj), constr)
        self.optval = prob.solve()
        self.x = x.value
        self.u = u.value
        return prob


    def test_mpc_quad(self):
        prob = self.mpc_setup(objective='quad')
        self._test_prob(prob)
        self._run_codegen_test(prob, '_test_mpc_quad')

    def _test_mpc_quad(self):
        self.mpc_setup(objective='quad')
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve(A=self.A, B=self.B, x0=self.x0)
        self.assertEqual(stats['status'], 'optimal')
        self.assertAlmostEqual(stats['objective'], self.optval)
        self.assertAlmostEqualMatrices(self.x, vars['x'])
        self.assertAlmostEqualMatrices(self.u, vars['u'])



    def test_mpc_1norm(self):
        prob = self.mpc_setup(objective='1norm')
        self._test_prob(prob)
        self._run_codegen_test(prob, '_test_mpc_1norm')

    def _test_mpc_1norm(self):
        self.mpc_setup(objective='1norm')
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve(A=self.A, B=self.B, x0=self.x0)
        self.assertEqual(stats['status'], 'optimal')
        self.assertAlmostEqual(stats['objective'], self.optval)
        self.assertAlmostEqualMatrices(self.x, vars['x'])
        self.assertAlmostEqualMatrices(self.u, vars['u'])




if __name__ == '__main__':
    unittest.main()
