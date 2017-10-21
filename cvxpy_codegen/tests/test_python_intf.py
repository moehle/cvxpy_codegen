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

MODULE = 'cvxpy_codegen.tests.test_python_intf'



class TestPythonIntf(tu.CodegenTestCase):
    class_name = 'TestPythonIntf'
    module = MODULE


    def test_infeas(self, atom=None):
        x = cvx.Variable(name='x')
        constr = [x >= 1, x <= 0]
        obj = x
        self.prob = cvx.Problem(cvx.Minimize(obj), constr)
        self._run_codegen_test(self.prob, '_test_infeas')

    def _test_infeas(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve()
        self.assertEqual(stats['status'], 'infeasible')
        self.assertAlmostEqual(stats['objective'], np.inf)



    def test_unbounded(self):
        x = cvx.Variable(name='x')
        constr = [x <= 1]
        obj = x
        self.prob = cvx.Problem(cvx.Minimize(obj), constr)
        self._run_codegen_test(self.prob, '_test_unbounded')

    def _test_unbounded(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve()
        self.assertEqual(stats['status'], 'unbounded')
        self.assertAlmostEqual(stats['objective'], -np.inf)



    def test_optimal(self):
        x = cvx.Variable(name='x')
        constr = [x >= 1]
        obj = x
        self.prob = cvx.Problem(cvx.Minimize(obj), constr)
        self._run_codegen_test(self.prob, '_test_optimal')

    def _test_optimal(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve()
        self.assertEqual(stats['status'], 'optimal')
        self.assertAlmostEqual(stats['objective'], 1.0)



    def test_scalars(self):
        x = cvx.Variable(name='x')
        p = cvx.Parameter(name='p')
        constr = [x >= p]
        obj = x
        self.prob = cvx.Problem(cvx.Minimize(obj), constr)
        self._run_codegen_test(self.prob, '_test_scalars')

    def _test_scalars(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve(p=2.0)
        self.assertEqual(stats['status'], 'optimal')
        self.assertAlmostEqual(stats['objective'], 2.0)
        vars, stats = cg_solve(p=2.0)

        with self.assertRaises(Exception) as cm:
            vars, stats = cg_solve(p=randn(1,1))
        self.assertEqual(str(cm.exception), 
                        "Parameter p should be an "
                        "int or float, but instead has "
                        "type <type 'numpy.ndarray'>")




    def test_vectors(self):
        x = cvx.Variable(10, name='x')
        p = cvx.Parameter(10, name='p')
        constr = [x >= p]
        obj = cvx.sum(x)
        self.prob = cvx.Problem(cvx.Minimize(obj), constr)
        self._run_codegen_test(self.prob, '_test_vectors')

    def _test_vectors(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve(np.ones((10,)))
        self.assertEqual(stats['status'], 'optimal')
        self.assertAlmostEqual(stats['objective'], 10.0)

        with self.assertRaises(Exception) as cm:
            vars, stats = cg_solve(p=randn(10,1))
        self.assertEqual(str(cm.exception), 
                        "Parameter p should have shape (10,), "
                        "but has shape (10, 1)")




    def test_matrices(self):
        x = cvx.Variable((10,5), name='x')
        p = cvx.Parameter((10,5), name='p')
        constr = [x >= p]
        obj = cvx.sum(x)
        self.prob = cvx.Problem(cvx.Minimize(obj), constr)
        self._run_codegen_test(self.prob, '_test_matrices')

    def _test_matrices(self):
        from cvxpy_codegen_solver import cg_solve
        vars, stats = cg_solve(np.ones((10,5)))
        self.assertEqual(stats['status'], 'optimal')
        self.assertAlmostEqual(stats['objective'], 50.0)

        with self.assertRaises(Exception) as cm:
            vars, stats = cg_solve(p=randn(10,))
        self.assertEqual(str(cm.exception), 
                        "Parameter p should have shape (10, 5), "
                        "but has shape (10,)")




if __name__ == '__main__':
    unittest.main()
