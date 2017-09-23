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

MODULE = 'cvxpy_codegen.tests.test_examples'


class TestMpc(tu.CodegenTestCase):
    class_name = 'TestMpc'

    def _test_mpc(self, objective='1norm'):
        np.random.seed(0)
        n = 2
        m = 2
        T = 3
        self.A_val = np.eye(n) + .2*np.random.randn(n,n)
        self.B_val = 5*np.random.randn(n,m)
        self.x0_val = 5*np.random.randn(n)

        A  = cvx.Parameter((n, n), name='A')
        B  = cvx.Parameter((n, m), name='B')
        x0 = cvx.Parameter(n, name='x0')

        A  = np.random.randn(n, n)
        B  = np.random.randn(n, m)
        x0 = np.random.randn(n)

        #A.value  = self.A_val
        #B.value  = self.B_val
        #x0.value = self.x0_val

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
        self._test_prob(prob)






    #def test_mpc_quad(self):
    #    self._test_mpc(objective='quad')

    def test_mpc_1norm(self):
        self._test_mpc(objective='1norm')

    #def test_mpc_exp1norm(self):
    #    self._test_mpc(objective='exp1norm')
    




class TestLeastSquares(tu.CodegenTestCase):
    class_name = 'TestLeastSquares'

    def test_least_squares(self):
        np.random.seed(0)
        n = 5
        m = 10
        self.A_val = np.random.randn(m,n)
        self.b_val = np.random.randn(m)
        A = cvx.Parameter((m, n), name='A', value=self.A_val)
        b = cvx.Parameter(m, name='b', value=self.b_val)
        x = cvx.Variable(n, name='x')
        objective = cvx.norm(A*x - b)

        self.prob = cvx.Problem(cvx.Minimize(objective))
        self._test_prob(self.prob)
    

    

    



if __name__ == '__main__':
    unittest.main()
