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
import numpy as np
import scipy.sparse as sp
import numpy as np
from cvxpy_codegen.object_data.param_data import ParamData
from cvxpy_codegen.object_data.var_data import VarData
from cvxpy_codegen import codegen
from cvxpy_codegen.tests.utils import TARGET_DIR
np.random.seed(0)
 


class TestCodegen(tu.CodegenTestCase):
    EPS = 1e-5


    def test_duplicate_var_names(self):
        x1 = cvx.Variable(name='x')
        x2 = cvx.Variable(name='x')
        prob = cvx.Problem(cvx.Minimize(x1 + x2), [])
        with self.assertRaises(Exception) as cm:
            codegen(prob, TARGET_DIR)
        self.assertEqual(str(cm.exception), 'Duplicate variable name "x".')


    def test_duplicate_param_names(self):
        p1 = cvx.Parameter(name='p')
        p2 = cvx.Parameter(name='p')
        x = cvx.Variable(name='x')
        prob = cvx.Problem(cvx.Minimize((p1 + p2)*x), [])
        with self.assertRaises(Exception) as cm:
            codegen(prob, TARGET_DIR)
        self.assertEqual(str(cm.exception), 'Duplicate parameter name "p".')


    def test_wrong_solver(self):
        x = cvx.Variable(name='x')
        prob = cvx.Problem(cvx.Minimize(x), [])
        with self.assertRaises(Exception) as cm:
            codegen(prob, TARGET_DIR, solver='SCS')
        self.assertEqual(str(cm.exception), 'Unknown solver SCS.')








if __name__ == '__main__':
    unittest.main()
