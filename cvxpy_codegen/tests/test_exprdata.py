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
import scipy.sparse as sp
import numpy as np
from cvxpy_codegen.param.expr_data import ParamData, CONST_ID
 


class TestParamData(tu.CodegenTestCase):
    EPS = 1e-5


    def test_param_data(self):
        n = 10
        m = 5
        A = cg.Parameter(m,n, name='A')

        A_paramdata = ParamData(A)

        # Sparsity pattern.
        A_sparsity = sp.csr_matrix(np.full((m,n), True, dtype=bool))
        self.assertEqualMatrices(A_paramdata.sparsity, A_sparsity)

        # Value
        self.assertEqualLists(A_paramdata.value.shape, (m,n))
        A_value = np.random.randn(m,n)
        A.value = A_value
        A_paramdata = ParamData(A)
        self.assertEqualMatrices(A_paramdata.value, A_value)

        # Other attributes
        self.assertEqual(A_paramdata.args, [])
        self.assertEqual(A_paramdata.size, (m,n))
        self.assertEqual(A_paramdata.length, m*n)
        self.assertEqual(A_paramdata.type, 'param')
        self.assertEqual(A_paramdata.name, 'A')
        self.assertEqual(A_paramdata.var_ids, [CONST_ID])
        self.assertEqual(A_paramdata.mem_name, 'A')







if __name__ == '__main__':
    unittest.main()
