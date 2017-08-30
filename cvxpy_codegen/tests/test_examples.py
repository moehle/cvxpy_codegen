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

    def mpc_setup(self, objective='quad'):
        np.random.seed(0)
        n = 3
        m = 1
        T = 6
        self.A_val = np.eye(n) + .2*np.random.randn(n,n)
        self.B_val = 5*np.random.randn(n,m)
        self.x0_val = 5*np.random.randn(n,1)

        A  = cvx.Parameter(n, n, name='A')
        B  = cvx.Parameter(n, m, name='B')
        x0 = cvx.Parameter(n, 1, name='x0')

        A.value  = self.A_val
        B.value  = self.B_val
        x0.value = self.x0_val

        x = cvx.Variable(n, T+1, name='x')
        u = cvx.Variable(m, T, name='u')

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

        self.prob = cvx.Problem(cvx.Minimize(obj), constr)
        self.prob.solve()
        self.x_opt = x.value


    def test_mpc_quad(self):
        test_name = '_test_mpc_quad'
        self.mpc_setup(objective='quad')
        self._run_codegen_test(self.prob, MODULE, self.class_name, test_name)

    def _test_mpc_quad(self):
        self.mpc_setup(objective='quad')
        from cvxpy_codegen_solver import cg_solve
        var_dict, stats_dict = cg_solve(x0=self.x0_val, A=self.A_val, B=self.B_val)
        self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
    

    def test_mpc_1norm(self):
        test_name = '_test_mpc_1norm'
        self.mpc_setup(objective='1norm')
        self._run_codegen_test(self.prob, MODULE, self.class_name, test_name)

    def _test_mpc_1norm(self):
        self.mpc_setup(objective='1norm')
        from cvxpy_codegen_solver import cg_solve
        var_dict, stats_dict = cg_solve(x0=self.x0_val, A=self.A_val, B=self.B_val)
        self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
    

    # TODO FAILS, probably due to ECOS..
    #def test_mpc_exp1norm(self):
    #    test_name = '_test_mpc_exp1norm'
    #    self.mpc_setup(objective='exp1norm')
    #    self.run_codegen_test(self.prob, MODULE, self.class_name, test_name)

    #def _test_mpc_exp1norm(self):
    #    self.mpc_setup(objective='exp1norm')
    #    from cvxpy_codegen_solver import cg_solve
    #    var_dict, stats_dict = cg_solve(x0=self.x0_val, A=self.A_val, B=self.B_val)
    #    self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
    




class TestLeastSquares(tu.CodegenTestCase):
    class_name = 'TestLeastSquares'

    def mpc_setup(self):
        np.random.seed(0)
        n = 5
        m = 10
        self.A_val = np.random.randn(m,n)
        self.b_val = np.random.randn(m,1)
        A = cvx.Parameter(m, n, name='A', value=self.A_val)
        b = cvx.Parameter(m, 1, name='b', value=self.b_val)
        x = cvx.Variable(n, name='x')
        objective = cvx.norm(A*x - b)

        self.prob = cvx.Problem(cvx.Minimize(objective))
        self.prob.solve()
        self.x_opt = x.value


    def test_least_squares(self):
        test_name = '_test_least_squares'
        self.mpc_setup()
        self._run_codegen_test(self.prob, MODULE, self.class_name, test_name)

    def _test_least_squares(self):
        self.mpc_setup()
        from cvxpy_codegen_solver import cg_solve
        var_dict, stats_dict = cg_solve(A=self.A_val, b=self.b_val)
        self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
    

    

    



if __name__ == '__main__':
    unittest.main()
