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

MODULE = 'cvxpy_codegen.tests.examples.test_markowitz'


class TestMarkowitzPortfolio(tu.CodegenTestCase):
    class_name = 'TestMarkowitzPortfolio'
    module = MODULE

    def setup_markowitz_portfolio(self):
        np.random.seed(0)
        n = 1000
        m = 50

        F = cvx.Parameter((m, n), name='F', value=randn(m, n))
        Dhalf = cvx.Parameter(n, name='Dhalf', value=randn(n))
        r = cvx.Parameter(n, name='r', value=randn(n))
        x = cvx.Variable(n, name='x')

        #objective = r.T*x + cvx.quad_form(x, cvx.diag(D)) + cvx.sum_squares(F*x)
        objective = r.T*x + cvx.sum_squares(Dhalf*x) + cvx.sum_squares(F*x)
        constrs = [x >= 0, cvx.sum(x) == 1]

        prob = cvx.Problem(cvx.Minimize(objective), constrs)
        self.optval = prob.solve(solver="ECOS")
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
