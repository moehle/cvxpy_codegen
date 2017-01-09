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
import cvxpy_codegen.tests.test_utils as tu
import cvxpy_codegen as cg
import numpy as np

MODULE = 'cvxpy_codegen.tests.test_param_handler'




class TestParamHandler(tu.CodegenTestCase):
    class_name = 'TestParamHandler'


    def atom_setup(self, first_atom='inplace', second_atom='inplace'):
        np.random.seed(0)
        n = 10

        A  = cg.Parameter(n, n, name='A', value = np.random.randn(n,n))
        B  = cg.Parameter(n, n, name='B', value = np.random.randn(n,n))
        C  = cg.Parameter(n, n, name='C', value = np.random.randn(n,n))


        m = 5
        n = 10
        p = 7
        const_mn = np.random.randn(m,n)
        A = cg.Parameter(m,n, name='A', value=np.random.randn(m,n))
        B = cg.Parameter(n,p, name='B', value=np.random.randn(n,p))
        C = cg.Parameter(m,p, name='C', value=np.random.randn(m,p))
        D = cg.Parameter(p,n, name='D', value=np.random.randn(p,n))
        P = cg.Parameter(n,n, name='P', value=np.random.randn(n,n))
        r = cg.Parameter(1,n, name='r', value=np.random.randn(1,n))
        c = cg.Parameter(n,1, name='c', value=np.random.randn(n,1))
        params = [A, B, C, D, P, r, c]
        self.params = dict([(p.name(), p.value) for p in params])

        result0 = self.get_expr(args,    first_atom)
        result1 = self.get_expr(result0, second_atom)

        x = cg.Variable(result1.size[0], result1.size[1], name='x')
        constr = [result1 == x]

        self.prob = cg.Problem(cg.Minimize(0), constr)
        self.prob.solve(verbose=True)
        self.x_opt = x.value
    
        if   first_atom == 'inplace':
            expr = -A
        elif first_atom == 'constant':
            result = -arg[3]
        elif first_atom == 'expr':
            result = arg[0] * arg[1]
        elif first_atom == 'param':
            result = arg[0]
        elif first_atom == 'varargs':
            result = arg[0] + arg[1] + arg[2]

        if   second_atom == 'inplace':
            result = -arg[0]
        elif second_atom == 'constant':
            result = -arg[3]
        elif second_atom == 'expr':
            result = arg[0] * arg[1]
        elif second_atom == 'param':
            result = arg[0]
        elif second_atom == 'varargs':
            result = arg[0] + arg[1] + arg[2]

# 
# 
#     ##################################
#     #  TEST VARARGS ATOMS            #
#     ##################################
# 
#     #def test_inplace_and_varargs(self):
#     #    test_name = '_test_inplace_and_varargs'
#     #    self.atom_setup(first_atom = 'inplace', second_atom = 'varargs')
#     #    self.run_codegen_test(self.prob, MODULE, self.class_name, test_name)
# 
#     #def _test_inplace_and_varargs(self):
#     #    self.atom_setup(first_atom = 'inplace', second_atom = 'varargs')
#     #    from cvxpy_codegen_solver import cg_solve
#     #    var_dict = cg_solve(A=self.A_val, B=self.B_val, C=self.C_val)
#     #    self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
# 
# 
#     #def test_constant_and_varargs(self):
#     #    test_name = '_test_constant_and_varargs'
#     #    self.atom_setup(first_atom = 'constant', second_atom = 'varargs')
#     #    self.run_codegen_test(self.prob, MODULE, self.class_name, test_name)
# 
#     #def _test_constant_and_varargs(self):
#     #    self.atom_setup(first_atom = 'constant', second_atom = 'varargs')
#     #    from cvxpy_codegen_solver import cg_solve
#     #    var_dict = cg_solve(B=self.B_val, C=self.C_val)
#     #    self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
# 
# 
#     #def test_expr_and_varargs(self):
#     #    test_name = '_test_expr_and_varargs'
#     #    self.atom_setup(first_atom = 'expr', second_atom = 'varargs')
#     #    self.run_codegen_test(self.prob, MODULE, self.class_name, test_name)
# 
#     #def _test_expr_and_varargs(self):
#     #    self.atom_setup(first_atom = 'expr', second_atom = 'varargs')
#     #    from cvxpy_codegen_solver import cg_solve
#     #    var_dict = cg_solve(A=self.A_val, B=self.B_val, C=self.C_val)
#     #    self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
# 
# 
#     def test_atoms(self, first_atom, second_atom):
#         method_name = '_test_' + first_atom + '_and_' + second_atom
#         self.atom_setup(first_atom = first_atom, second_atom = second_atom)
#         self.run_codegen_test(self.prob, MODULE, self.class_name, method_name)
# 
# 
#     def _test_atoms(self, first_atom, second_atom):
#         method_name = '_test_' + first_atom + '_and_' + second_atom
#         self.atom_setup(first_atom = first_atom, second_atom = second_atom)
#         from cvxpy_codegen_solver import cg_solve
#         param_names = list(inspect.signature(cg_solve).parameters.keys())
#         params = dict()
#         for p in param_names:
#             params[p] = self.params[p]
# 
#         var_dict = cg_solve(**params)
#         print(self.x_opt)
#         print(var_dict['x'])
#         self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
# 
# 
#     def test_param_and_varargs(self):
# 
#     def _test_param_and_varargs(self):
#         self.atom_setup(first_atom = 'param', second_atom = 'varargs')
#         from cvxpy_codegen_solver import cg_solve
#         var_dict = cg_solve(A=self.A_val, B=self.B_val, C=self.C_val)
#         self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
#
#
#
#    ###################################
#    ##  TEST NORMAL ATOMS             #
#    ###################################
#
#    #test_inplace_and_expr():
#
#    #test_constant_and_expr():
#
#    #test_varargs_and_expr():
#
#
#    ###################################
#    ##  TEST INPLACE ATOMS            #
#    ###################################
#
#    #test_expr_and_inplace():
#
#    #test_constant_and_inplace():
#
#    #test_varargs_and_inplace():


if __name__ == '__main__':
    unittest.main()
