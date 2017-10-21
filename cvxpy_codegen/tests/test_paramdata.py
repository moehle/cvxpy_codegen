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
np.random.seed(0)
 


class TestParamData(tu.CodegenTestCase):
    EPS = 1e-5


    def setUp(self):
        self.n = 3
        self.m = 2

        self.B_value = np.random.randn(self.m,self.n)
        self.d_value = np.random.randn(self.n)
        self.f_value = np.random.randn()

        A = cvx.Parameter((self.m,self.n), name='A')
        B = cvx.Parameter((self.m,self.n), name='B', value=self.B_value)
        c = cvx.Parameter((self.n,), name='c')
        d = cvx.Parameter((self.n,), name='d', value=self.d_value)
        e = cvx.Parameter((), name='e')
        f = cvx.Parameter((), name='f', value=self.f_value)

        self.A_paramdata = ParamData(A)
        self.B_paramdata = ParamData(B)
        self.c_paramdata = ParamData(c)
        self.d_paramdata = ParamData(d)
        self.e_paramdata = ParamData(e)
        self.f_paramdata = ParamData(f)



    def test_sparsity(self):

        A_sparsity = sp.csr_matrix(np.full((self.m,self.n), True, dtype=bool))
        self.assertEqualMatrices(self.A_paramdata.sparsity, A_sparsity)

        c_sparsity = sp.csr_matrix(np.full((self.n,1), True, dtype=bool))
        self.assertEqualMatrices(self.c_paramdata.sparsity, c_sparsity)

        e_sparsity = sp.csr_matrix(np.full((1,1), True, dtype=bool))
        self.assertEqualMatrices(self.e_paramdata.sparsity, e_sparsity)


    def test_value(self):

        self.assertEqualLists(self.A_paramdata.value.shape, (self.m,self.n))
        self.assertEqualLists(self.B_paramdata.value.shape, (self.m,self.n))
        self.assertEqualMatrices(self.B_paramdata.value, self.B_value)

        self.assertEqualLists(self.c_paramdata.value.shape, (self.n,))
        self.assertEqualLists(self.d_paramdata.value.shape, (self.n,))
        self.assertEqualMatrices(self.d_paramdata.value, self.d_value)

        self.assertEqual(self.f_paramdata.value, self.f_value)


    def test_misc(self):

        self.assertEqual(self.A_paramdata.args, [])
        self.assertEqual(self.A_paramdata.shape, (self.m,self.n))
        self.assertEqual(self.A_paramdata.length, self.m*self.n)
        self.assertEqual(self.A_paramdata.type, 'param')
        self.assertEqual(self.A_paramdata.name, 'A')
        self.assertEqual(self.A_paramdata.var_ids, [])
        self.assertEqual(self.A_paramdata.mem_name, 'A')


    def test_c_print(self):
        s = '    params.B[0][0] = %f;\n' % self.B_value[0,0] + \
            '    params.B[0][1] = %f;\n' % self.B_value[0,1] + \
            '    params.B[0][2] = %f;\n' % self.B_value[0,2] + \
            '    params.B[1][0] = %f;\n' % self.B_value[1,0] + \
            '    params.B[1][1] = %f;\n' % self.B_value[1,1] + \
            '    params.B[1][2] = %f;\n' % self.B_value[1,2] + \
            '    \n'
        self.assertEqual(s, self.B_paramdata.c_print())

        s = '    params.d[0] = %f;\n' % self.d_value[0] + \
            '    params.d[1] = %f;\n' % self.d_value[1] + \
            '    params.d[2] = %f;\n' % self.d_value[2] + \
            '    \n'
        self.assertEqual(s, self.d_paramdata.c_print())

        s = '    params.f = %f;\n' % self.f_value + \
            '    \n'
        self.assertEqual(s, self.f_paramdata.c_print())




    def test_c_print_struct(self):

        s = '    double A[2][3];\n'
        self.assertEqual(s, self.A_paramdata.c_print_struct())

        s = '    double c[3];\n'
        self.assertEqual(s, self.c_paramdata.c_print_struct())

        s = '    double e;\n'
        self.assertEqual(s, self.e_paramdata.c_print_struct())




if __name__ == '__main__':
    unittest.main()
