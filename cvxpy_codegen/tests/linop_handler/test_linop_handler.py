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

import sys
import os
import subprocess
import unittest
import cvxpy_codegen.tests.utils as tu
import cvxpy_codegen as cg
import numpy as np
import json
from cvxpy_codegen.linop_sym.linop_handler_sym import LinOpHandlerSym
from jinja2 import Environment, PackageLoader, contextfilter
from cvxpy_codegen.utils.utils import render, make_target_dir
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.problems.solvers.ecos_intf import ECOS
import cvxpy
from cvxpy.problems.problem_data.sym_data import SymData
from cvxpy_codegen.code_generator import CodeGenerator

ECOS = ECOS()

HARNESS_C = 'tests/linop_handler/harness.c.jinja'
CODEGEN_H = 'tests/linop_handler/codegen.h.jinja'
target_dir = tu.TARGET_DIR


class TestLinopHandler(tu.CodegenTestCase):

    # Define some Parameters and Constants to build tests with.
    def setUp(self):
        np.random.seed(0)
        m = 10
        n = 20
        p = 15
        self.m = m
        self.n = n
        self.p = p
        self.var_mn = cg.Variable(m, n, name='var_mn')
        self.var_np = cg.Variable(n, p, name='var_np')
        self.var_mp = cg.Variable(m, p, name='var_mp')
        self.var_pn = cg.Variable(p, n, name='var_pn')
        self.var_nn = cg.Variable(n, n, name='var_nn')
        self.var_n1 = cg.Variable(n, 1, name='var_n1')
        self.var_1n = cg.Variable(1, n, name='var_1n')
        self.var_11 = cg.Variable(1, 1, name='var_11')
        self.param_mn = cg.Parameter(m, n, name='param_mn', value=np.random.randn(m, n))
        self.param_np = cg.Parameter(n, p, name='param_np', value=np.random.randn(n, p))
        self.param_mp = cg.Parameter(m, p, name='param_mp', value=np.random.randn(m, p))
        self.param_pn = cg.Parameter(p, n, name='param_pn', value=np.random.randn(p, n))
        self.param_nn = cg.Parameter(n, n, name='param_nn', value=np.random.randn(n, n))
        self.param_n1 = cg.Parameter(n, 1, name='param_n1', value=np.random.randn(n, 1))
        self.param_1n = cg.Parameter(1, n, name='param_1n', value=np.random.randn(1, n))
        self.param_11 = cg.Parameter(1, 1, name='param_11', value=np.random.randn(1, 1))
        self.const_mn = np.random.randn(m, n)
        self.const_np = np.random.randn(n, p)
        self.const_mp = np.random.randn(m, p)
        self.const_pn = np.random.randn(p, n)
        self.const_nn = np.random.randn(n, n)
        self.const_n1 = np.random.randn(n, 1)
        self.const_1n = np.random.randn(1, n)
        self.const_11 = np.random.randn(1, 1)


    # Test atoms (exprs be convex scalars, involving a variable):

    def test_sum_entries(self):
        self._test_expr(cg.sum_entries(self.var_mn))
        self._test_expr(cg.sum_entries(self.var_mn + self.param_mn))
        self._test_expr(cg.sum_entries(self.var_mn + self.const_mn))
    
    def test_mul(self):
        self._test_expr(cg.sum_entries(self.var_mn * self.param_np))
        self._test_expr(cg.sum_entries(self.var_mn * self.const_np))
        self._test_expr(cg.sum_entries(self.param_mn * self.var_np))
        self._test_expr(cg.sum_entries(self.const_mn * self.var_np))
    
    def test_mul_elemwise(self):
        self._test_expr(cg.sum_entries(cg.mul_elemwise(self.param_mn, self.var_mn)))
        self._test_expr(cg.sum_entries(cg.mul_elemwise(self.const_mn, self.var_mn)))
    
    def test_mul_elem(self):
        self._test_expr(cg.sum_entries(self.const_np * self.var_11))
        self._test_expr(cg.sum_entries(self.var_11 * self.const_np))
        self._test_expr(cg.sum_entries(self.var_mn * self.const_11))
        self._test_expr(cg.sum_entries(self.var_11 * self.param_np))
        self._test_expr(cg.sum_entries(self.var_mn * self.param_11))
    
    def test_div_elem(self):
        self._test_expr(cg.sum_entries(self.var_11 / self.const_11))
        self._test_expr(cg.sum_entries(self.var_mn / self.const_11))
        self._test_expr(cg.sum_entries(self.var_11 / self.param_11))
        self._test_expr(cg.sum_entries(self.var_mn / self.param_11))
    
    def test_neg(self):
        self._test_expr(cg.sum_entries(-self.var_mn))
        self._test_expr(cg.sum_entries(self.param_mn - self.var_mn))
        self._test_expr(cg.sum_entries(self.const_mn - self.var_mn))
    
    def test_trace(self):
        self._test_expr(cg.trace(self.var_nn))
        self._test_expr(cg.trace(self.var_11))
    
    def test_transpose(self):
        self._test_expr(cg.sum_entries(cg.trace(self.var_mn)))
    
    def test_reshape(self):
        self._test_expr(cg.sum_entries(cg.reshape(self.var_mn, self.n, self.m)))
    
    def test_index(self):
        self._test_expr(cg.sum_entries(self.var_mn[2:13:2,2:8]))
        self._test_expr(cg.sum_entries(self.var_n1[2:13:2]))
    
    def test_diff(self):
        self._test_expr(cg.sum_entries(cg.diff(self.var_n1)))
        self._test_expr(cg.sum_entries(cg.diff(self.var_mn)))
    
    def test_hstack(self):
        self._test_expr(cg.sum_entries(
                cg.hstack(self.var_n1, self.var_np, self.var_nn)))
        self._test_expr(cg.sum_entries(
                cg.hstack(self.var_n1, self.const_np, self.param_nn)))
    
    def test_vstack(self):
        self._test_expr(cg.sum_entries(
                cg.vstack(self.var_1n, self.var_mn, self.var_nn)))
        self._test_expr(cg.sum_entries(
                cg.vstack(self.var_1n, self.const_mn, self.param_nn)))
    
    def test_transpose(self):
        self._test_expr(cg.sum_entries(self.var_mn.T))
    
    def test_upper_tri(self):
        self._test_expr(cg.sum_entries(self.var_nn))
        self._test_expr(cg.sum_entries(self.var_nn + self.const_nn))
        self._test_expr(cg.sum_entries(self.var_nn + self.param_nn))
    
    

    # Test combinations of linop_handler and param_handler:
    
    def test_combination(self):
        self._test_expr(cg.sum_entries((self.param_mn * self.param_nn) * self.var_n1))
        self._test_expr(cg.sum_entries((self.param_mn * self.const_nn) * self.var_n1))
        self._test_expr(cg.sum_entries(-(self.param_mn * self.param_nn) * self.var_n1))
        self._test_expr(cg.sum_entries(-(self.param_mn * self.const_nn) * self.var_n1))




    def _test_expr(self, expr, printing=False):
        # Get Callback param.
        prob = cg.Problem(cvxpy.Minimize(expr))
        obj, constrs = expr.canonicalize()
        data = ECOS.get_problem_data(obj, constrs, prob._cached_data)
        true_obj_coeff  = data[s.C]
        true_obj_offset = data[s.OFFSET]
        true_eq_coeff   = data[s.A]
        true_eq_offset  = data[s.B]
        true_leq_coeff  = data[s.G]
        true_leq_offset = data[s.H]

        # Do code generation
        vars = prob.variables()
        params = prob.parameters()
        obj, constraints = prob.canonical_form
        code_generator = CodeGenerator(obj, constraints, vars, params)
        code_generator.codegen(target_dir)
        template_vars = code_generator.template_vars

        # Set up test harness.
        render(target_dir, template_vars, HARNESS_C, 'harness.c')
        render(target_dir, template_vars, CODEGEN_H, 'codegen.h')
        test_data = self._run_test(target_dir)
        test_obj_coeff  = np.array(test_data['obj_coeff'])
        test_obj_offset = np.array(test_data['obj_offset'])
        test_eq_coeff  = sp.csc_matrix((test_data['eq_nzval'],
                                        test_data['eq_rowidx'],
                                        test_data['eq_colptr']),
                                        shape = (test_data['eq_size0'],
                                                 test_data['eq_size1']))
        test_eq_offset = np.array(test_data['eq_offset'])
        test_leq_coeff = sp.csc_matrix((test_data['eq_nzval'],
                                        test_data['eq_rowidx'],
                                        test_data['eq_colptr']),
                                        shape = (test_data['eq_size0'],
                                                 test_data['eq_size1']))
        test_leq_offset = np.array(test_data['leq_offset'])

        if printing:
            print('\nTest objective coeff  :\n',   test_obj_coeff)
            print('\nTrue objective coeff  :\n',   true_obj_coeff)

            print('\nTest objective offset :\n',   test_obj_offset)
            print('\nTrue objective offset :\n',   true_obj_offset)

            print('\nTest equality offset :\n',    test_eq_coeff)
            print('\nTrue equality offset :\n',    true_eq_coeff)

            print('\nTest equality offset :\n',    test_eq_offset)
            print('\nTrue equality offset :\n',    true_eq_offset)

            print('\nTest inequality offset :\n',  test_leq_coeff)
            print('\nTrue inequality offset :\n',  true_leq_coeff)

            print('\nTest inequality offset :\n',  test_leq_offset)
            print('\nTrue inequality offset :\n',  true_leq_offset)

        self.assertAlmostEqualMatrices(true_obj_coeff,  test_obj_coeff)
        self.assertAlmostEqualMatrices(true_obj_offset, test_obj_offset)
        self.assertAlmostEqualMatrices(true_eq_coeff,   test_eq_coeff)
        self.assertAlmostEqualMatrices(true_eq_offset,  test_eq_offset)
        self.assertAlmostEqualMatrices(true_leq_coeff,  test_leq_coeff)
        self.assertAlmostEqualMatrices(true_leq_offset, test_leq_offset)



    def _run_test(self, target_dir):
        prev_path = os.getcwd()
        os.chdir(target_dir)
        output = subprocess.check_output(
                         ['gcc', 'harness.c', 'linop.c', 'param.c', '-o' 'main'],
                         stderr=subprocess.STDOUT)
        exec_output = subprocess.check_output(['./main'], stderr=subprocess.STDOUT)
        os.chdir(prev_path)
        return json.loads(exec_output.decode("utf-8"))
        


if __name__ == '__main__':
   unittest.main()
