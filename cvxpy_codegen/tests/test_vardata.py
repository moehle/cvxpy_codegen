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
 


class TestVarData(tu.CodegenTestCase):
    EPS = 1e-5

    def setUp(self):
        self.n = 3
        self.m = 2

        self.X = cvx.Variable((self.m, self.n), name='X')
        self.y = cvx.Variable((self.n,), name='y')
        self.z = cvx.Variable(name='z')

        self.X_vardata = VarData(self.X, 0)
        self.y_vardata = VarData(self.y, 0)
        self.z_vardata = VarData(self.z, 0)



    def test_attributes(self):
        self.assertEqual(self.X_vardata.args, [])
        self.assertEqual(self.X_vardata.shape, (self.m, self.n))
        self.assertEqual(self.X_vardata.length, self.m*self.n)
        self.assertEqual(self.X_vardata.name, 'X')
        self.assertEqual(self.X_vardata.var_ids, {self.X.id})




    def test_c_print(self):

        s = '    printf("X[0][0] = %f\\n", vars.X[0][0]);\n' + \
            '    printf("X[1][0] = %f\\n", vars.X[1][0]);\n' + \
            '    printf("X[0][1] = %f\\n", vars.X[0][1]);\n' + \
            '    printf("X[1][1] = %f\\n", vars.X[1][1]);\n' + \
            '    printf("X[0][2] = %f\\n", vars.X[0][2]);\n' + \
            '    printf("X[1][2] = %f\\n", vars.X[1][2]);\n' + \
            '    \n'
        self.assertEqual(s, self.X_vardata.c_print())

        s = '    printf("y[0] = %f\\n", vars.y[0]);\n' + \
            '    printf("y[1] = %f\\n", vars.y[1]);\n' + \
            '    printf("y[2] = %f\\n", vars.y[2]);\n' + \
            '    \n'
        self.assertEqual(s, self.y_vardata.c_print())

        s = '    printf("z = %f\\n", vars.z);\n' + \
            '    \n'
        self.assertEqual(s, self.z_vardata.c_print())




    def test_c_print_getvar(self):

        s = '    for(i=0; i<2; i++){\n' + \
            '        for(j=0; j<3; j++){\n' + \
            '            vars->X[i][j] = *(work->primal_var + 0 + i + 2*j);\n' + \
            '        }\n' + \
            '    }\n' + \
            '\n'
        self.assertEqual(s, self.X_vardata.c_print_getvar())

        s = '    for(i=0; i<3; i++){\n' + \
            '        vars->y[i] = *(work->primal_var + 0 + i);\n' + \
            '    }\n' + \
            '\n'
        self.assertEqual(s, self.y_vardata.c_print_getvar())

        s = '    vars->z = *(work->primal_var + 0);\n' + \
            '\n'
        self.assertEqual(s, self.z_vardata.c_print_getvar())





    def test_c_print_struct(self):

        s = '    double X[2][3];\n'
        self.assertEqual(s, self.X_vardata.c_print_struct())

        s = '    double y[3];\n'
        self.assertEqual(s, self.y_vardata.c_print_struct())

        s = '    double z;\n'
        self.assertEqual(s, self.z_vardata.c_print_struct())







if __name__ == '__main__':
    unittest.main()

