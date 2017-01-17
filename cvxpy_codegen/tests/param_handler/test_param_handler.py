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
from cvxpy_codegen.param.param_handler import ParamHandler
from jinja2 import Environment, PackageLoader, contextfilter
from cvxpy_codegen.utils.utils import render, make_target_dir
import scipy.sparse as sp

HARNESS_C = 'tests/param_handler/harness.c.jinja'
CODEGEN_H = 'tests/param_handler/codegen.h.jinja'
target_dir = tu.TARGET_DIR


class TestParamHandler(tu.CodegenTestCase):

    # Define some Parameters and Constants to build tests with.
    def setUp(self):
        np.random.seed(0)
        m = 10
        n = 20
        p = 15
        self.m = m
        self.n = n
        self.p = p
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


    # Test each atom:
    def test_abs(self):
        self._test_expr(cg.abs(self.param_mn))
        self._test_expr(cg.abs(self.param_mn + self.const_mn))
        self._test_expr(cg.abs(-self.param_mn))

    def test_add(self):
        self._test_expr(self.param_mn + self.param_mn)
        self._test_expr(self.param_mn - self.param_mn)
        self._test_expr(self.param_n1 + self.const_n1)
        self._test_expr(self.param_n1 - self.const_n1)
        self._test_expr(self.param_mn + self.param_11)
        self._test_expr(self.param_mn + self.param_11 + self.const_11)
        self._test_expr(self.param_11 + self.const_mn + self.const_mn)
        self._test_expr(self.const_mn + self.const_mn + self.param_11)
        self._test_expr(self.const_mn + self.const_11 + self.param_mn)

    def test_diag_vec(self):
        self._test_expr(cg.diag(self.param_n1))
        self._test_expr(cg.diag(-self.param_n1))
        self._test_expr(cg.diag(self.param_1n))

    def test_diag_mat(self):
        self._test_expr(cg.diag(self.param_nn))
        self._test_expr(cg.diag(cg.diag(self.param_n1)))
        self._test_expr(cg.diag(self.param_nn - cg.diag(self.param_n1)))
       
    def test_hstack(self):
        self._test_expr(cg.hstack(self.param_nn, self.param_n1))
        self._test_expr(cg.hstack(self.param_np, self.const_n1, self.param_nn))

    def test_index(self):
        self._test_expr(self.param_mn[0:8:2, 1:17:3])
        self._test_expr(self.param_n1[0:8:2])
        self._test_expr(self.param_1n[:5])
        self._test_expr(self.param_1n[5:])

    def test_max_entries(self):
        self._test_expr(cg.max_entries(self.param_mn))
        self._test_expr(cg.max_entries(-self.param_n1[:4]))

    def test_mul(self):
        self._test_expr(self.param_mn * self.param_np)
        self._test_expr(self.param_mn * self.const_np)
        self._test_expr(self.param_mn * self.param_n1)
        self._test_expr(self.param_mn * self.param_11)
        self._test_expr(self.param_11 * self.param_mn)
        self._test_expr(self.const_mn * self.param_11)
        self._test_expr(self.const_11 * self.param_mn)

    def test_mul_elemwise(self):
        self._test_expr(cg.mul_elemwise(self.param_mn, self.const_mn))
        self._test_expr(cg.mul_elemwise(self.param_mn, self.param_mn))
        self._test_expr(cg.mul_elemwise(self.param_n1, self.param_n1))
        self._test_expr(cg.mul_elemwise(self.param_n1, self.const_n1))
        self._test_expr(cg.mul_elemwise(self.param_11, self.param_11))
        self._test_expr(cg.mul_elemwise(self.param_11, self.const_11))

    def test_neg(self):
        self._test_expr(-self.param_mn)
        self._test_expr(-self.param_n1)

    def test_reshape(self):
        self._test_expr(cg.reshape(self.param_mn, self.n, self.m))
        self._test_expr(cg.reshape(self.param_mn + self.const_mn, self.n, self.m))
        self._test_expr(cg.reshape(self.param_n1, 1, self.n))

    def test_trace(self):
        self._test_expr(cg.trace(self.param_nn))
        self._test_expr(cg.trace(cg.diag(self.param_n1)))
        self._test_expr(cg.trace(self.param_11))

    def test_vstack(self):
        self._test_expr(cg.vstack(self.param_n1, self.param_11, self.const_11))
        self._test_expr(cg.vstack(self.param_mn, self.param_pn, self.const_1n))

    def test_const_expr(self):
        self._test_expr(cg.max_entries(self.const_nn) + self.param_11)
    



    def _test_expr(self, expr, printing=False):
        # Get Callback param.
        expr_canon = expr.canonicalize()[0]
        cb_param = expr_canon.data
        true_expr_value = sp.csc_matrix(cb_param.value)

        # Set up param handler.
        param_handler = ParamHandler(expr_canon, [], [])
        template_vars = param_handler.get_template_vars()
        make_target_dir(target_dir)
        param_handler.render(target_dir)

        # Set up test handler.
        render(target_dir, template_vars, HARNESS_C, 'harness.c')
        render(target_dir, template_vars, CODEGEN_H, 'codegen.h')
        tested_cb_param_values = self._run_test(target_dir)
        mat = list(tested_cb_param_values.values())[0]
        test_expr_value = sp.csc_matrix((mat['nzval'],
                                         mat['rowidx'],
                                         mat['colptr']))
        if printing:
            print('\nTrue value:\n', true_expr_value)
            print('\nTest value:\n', test_expr_value)
            print('\nDifference:\n', test_expr_value - true_expr_value)
        self.assertAlmostEqualMatrices(true_expr_value, test_expr_value)


    def _run_test(self, target_dir):
        prev_path = os.getcwd()
        os.chdir(target_dir)
        output = subprocess.check_output(
                         ['gcc', 'harness.c', 'param.c', '-o' 'main'],
                         stderr=subprocess.STDOUT)
        exec_output = subprocess.check_output(['./main'], stderr=subprocess.STDOUT)
        os.chdir(prev_path)
        return json.loads(exec_output.decode("utf-8"))
        


if __name__ == '__main__':
   unittest.main()
