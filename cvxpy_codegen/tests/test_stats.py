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
import cvxpy_codegen as cg
import numpy as np

MODULE = 'cvxpy_codegen.tests.test_stats'


class TestStats(tu.CodegenTestCase):
    class_name = 'TestStats'

    def test_infeas(self, atom=None):
        x = cg.Variable()
        constr = [x >= 1, x <= 0]
        obj = 0
        self.prob = cg.Problem(cg.Minimize(obj), constr)
        self._run_codegen_test(self.prob, MODULE, self.class_name, '_test_infeas')

    def _test_infeas(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve()
        self.assertEqual(stats['status'], 'infeasible')
        self.assertAlmostEqual(stats['objective'], np.inf)



    def test_unbounded(self):
        x = cg.Variable()
        constr = [x <= 1]
        obj = x
        self.prob = cg.Problem(cg.Minimize(obj), constr)
        self._run_codegen_test(self.prob, MODULE, self.class_name, '_test_unbounded')

    def _test_unbounded(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve()
        self.assertEqual(stats['status'], 'unbounded')
        self.assertAlmostEqual(stats['objective'], -np.inf)



    def test_optimal(self):
        x = cg.Variable()
        constr = [x >= 1]
        obj = x
        self.prob = cg.Problem(cg.Minimize(obj), constr)
        self._run_codegen_test(self.prob, MODULE, self.class_name, '_test_optimal')

    def _test_optimal(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve()
        self.assertEqual(stats['status'], 'optimal')
        self.assertAlmostEqual(stats['objective'], 1.0)




if __name__ == '__main__':
    unittest.main()
