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

MODULE = 'cvxpy_codegen.tests.test_examples'


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








class TestLeastSquares(tu.CodegenTestCase):
    class_name = 'TestLeastSquares'
    module = MODULE

    def setup_least_squares(self):
        np.random.seed(0)
        n = 5
        m = 10
        self.A = np.random.randn(m,n)
        self.b = np.random.randn(m)
        A = cvx.Parameter((m, n), name='A', value=self.A)
        b = cvx.Parameter(m, name='b', value=self.b)
        x = cvx.Variable(n, name='x')
        objective = cvx.norm(A*x - b)

        prob = cvx.Problem(cvx.Minimize(objective))
        self._test_prob(prob)
        self.optval = prob.solve()
        self.x = x.value
        return prob


    def test_least_squares(self):
        prob = self.setup_least_squares()
        self._test_prob(prob)
        self._run_codegen_test(prob, '_test_least_squares')

    def _test_least_squares(self):
        self.setup_least_squares()
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve(A=self.A, b=self.b)
        self.assertAlmostEqual(stats['objective'], self.optval)
        self.assertAlmostEqualMatrices(self.x, vars['x'])
    

   
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

  

class TestMarkowitzPortfolio(tu.CodegenTestCase):
    class_name = 'TestMarkowitzPortfolio'
    module = MODULE

    def setup_markowitz_portfolio(self):
        np.random.seed(0)
        n = 10
        m = 4

        F = cvx.Parameter((m, n), name='F', value=randn(m, n))
        Dhalf = cvx.Parameter(n, name='Dhalf', value=randn(n))
        r = cvx.Parameter(n, name='r', value=randn(n))
        x = cvx.Variable(n, name='x')

        #objective = r.T*x + cvx.quad_form(x, cvx.diag(D)) + cvx.sum_squares(F*x)
        objective = r.T*x + cvx.sum_squares(Dhalf*x) + cvx.sum_squares(F*x)
        constrs = [x >= 0, cvx.sum(x) == 1]

        prob = cvx.Problem(cvx.Minimize(objective), constrs)
        self.optval = prob.solve()
        self.x = x.value

        self.F = F.value
        self.Dhalf = Dhalf.value
        self.r = r.value

        return prob


    def test_markowitz_portfolio(self):
        prob = self.setup_markowitz_portfolio()
        self._test_prob(prob)
        self._run_codegen_test(prob, '_test_markowitz_portfolio')

    def _test_markowitz_portfolio(self):
        self.setup_markowitz_portfolio()
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve(F=self.F, Dhalf=self.Dhalf, r=self.r)
        self.assertAlmostEqual(stats['objective'], self.optval)
        self.assertAlmostEqualMatrices(self.x, vars['x'])
   

    



if __name__ == '__main__':
    unittest.main()
