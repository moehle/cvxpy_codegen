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
import cvxpy as cvx
import numpy as np
from numpy.random import randn
import json
from jinja2 import Environment, PackageLoader, contextfilter
from cvxpy_codegen.utils.utils import render, make_target_dir
import scipy.sparse as sp
from scipy.sparse import random
import cvxpy.settings as s
from cvxpy.problems.solvers.ecos_intf import ECOS
import cvxpy
from cvxpy.problems.problem_data.sym_data import SymData
from cvxpy_codegen import codegen

ECOS = ECOS()

HARNESS_C = 'tests/expr_handler/harness.c.jinja'
CODEGEN_H = 'tests/expr_handler/codegen.h.jinja'
target_dir = tu.TARGET_DIR



class TestExprHandler(tu.CodegenTestCase):

    # Define some Parameters and Constants to build tests with.
    def setUp(self):
        np.random.seed(0)
        m = 8
        n = 12
        p = 10
        self.m = m
        self.n = n
        self.p = p
        self.var_mn = cvx.Variable((m, n), name='var_mn')
        self.var_np = cvx.Variable((n, p), name='var_np')
        self.var_mp = cvx.Variable((m, p), name='var_mp')
        self.var_pn = cvx.Variable((p, n), name='var_pn')
        self.var_nn = cvx.Variable((n, n), name='var_nn')
        self.var_n1 = cvx.Variable((n, 1), name='var_n1')
        self.var_1n = cvx.Variable((1, n), name='var_1n')
        self.var_11 = cvx.Variable((1, 1), name='var_11')
        self.var_n  = cvx.Variable((n,),   name='var_n')
        self.var    = cvx.Variable((),     name='var')
        self.param_mn = cvx.Parameter((m, n), name='param_mn', value=randn(m, n))
        self.param_np = cvx.Parameter((n, p), name='param_np', value=randn(n, p))
        self.param_mp = cvx.Parameter((m, p), name='param_mp', value=randn(m, p))
        self.param_pn = cvx.Parameter((p, n), name='param_pn', value=randn(p, n))
        self.param_nn = cvx.Parameter((n, n), name='param_nn', value=randn(n, n))
        self.param_n1 = cvx.Parameter((n, 1), name='param_n1', value=randn(n, 1))
        self.param_1n = cvx.Parameter((1, n), name='param_1n', value=randn(1, n))
        self.param_11 = cvx.Parameter((1, 1), name='param_11', value=randn(1, 1))
        self.param_n  = cvx.Parameter((n,),   name='param_n',  value=randn(n))
        self.param    = cvx.Parameter((),     name='param',    value=randn())

        self.const_mn = random(m, n, .5, format='coo')
        self.const_np = random(n, p, .5, format='coo')
        self.const_mp = random(m, p, .5, format='coo')
        self.const_pn = random(p, n, .5, format='coo')
        self.const_nn = random(n, n, .5, format='coo')
        self.const_n1 = random(n, 1, .5, format='coo')
        self.const_1n = random(1, n, .5, format='coo')
        self.const_11 = random(1, 1, .5, format='coo')
        self.const_n  = randn(n,)
        self.const    = randn()
        
        self.const_mn = randn(m, n)
        self.const_np = randn(n, p)
        self.const_mp = randn(m, p)
        self.const_pn = randn(p, n)
        self.const_nn = randn(n, n)
        self.const_n1 = randn(n, 1)
        self.const_1n = randn(1, n)
        self.const_11 = randn(1, 1)
        self.const_n  = randn(n,)
        self.const    = randn()



    ###########################
    # FUNCTIONS OF PARAMETERS #
    ###########################

    #def test_abs(self):
    #    self._test_const_expr(cvx.abs(self.param_mn))
    #    self._test_const_expr(cvx.abs(self.const_mn))
    #    self._test_const_expr(cvx.abs(self.param_mn + self.const_mn))
    #    self._test_const_expr(cvx.abs(-self.param_mn))
    #    self._test_const_expr(cvx.abs(-self.param_n))
    #    self._test_const_expr(cvx.abs(-self.param))

    #def test_max(self):
    #    self._test_const_expr(cvx.max(self.param_mn))
    #    self._test_const_expr(cvx.max(-self.param_n1[:4]))


    def test_add(self):
        self._test_expr(self.param_mn + self.var_mn)
        self._test_expr(self.var_mn + self.var_mn)
        self._test_expr(self.var_mn + self.var + self.var_11)
        self._test_expr(self.const_mn - self.var_mn)
        self._test_expr(self.param_n1 + self.var_n1)
        self._test_expr(self.param_n1 - self.var_n1)
        self._test_expr(self.param_n + self.var_n)
        self._test_expr(self.param + self.var)
        self._test_expr(self.param_11 + self.var_mn + self.const_mn)
        self._test_expr(self.const_mn + self.var_mn + self.param_11)
        self._test_const_expr(self.param_11 + self.var_mn)
        self._test_const_expr(self.param_mn + self.var_11)
        self._test_const_expr(self.param_mn + self.var_11 + self.const_11)
        self._test_const_expr(self.param_11 + self.var_11 + self.const_11)
        self._test_const_expr(self.param_11 + self.var_11 + self.const_mn)
        self._test_const_expr(self.const_mn + self.var_11 + self.param_mn)
        self._test_const_expr(self.param_mn + self.param_mn)
        self._test_const_expr(self.param_mn - self.param_mn)
        self._test_const_expr(self.param_n1 + self.const_n1)
        self._test_const_expr(self.param_n1 - self.const_n1)
        self._test_const_expr(self.param_mn + self.param_11)
        self._test_const_expr(self.param_11 + self.const_mn)
        self._test_const_expr(self.param_mn + self.param_11 + self.const_11)
        self._test_const_expr(self.param_11 + self.param_11 + self.const_11)
        self._test_const_expr(self.param_11 + self.param_11 + self.const_mn)
        self._test_const_expr(self.param_11 + self.const_mn + self.const_mn)
        self._test_const_expr(self.const_mn + self.const_mn + self.param_11)
        self._test_const_expr(self.const_mn + self.const_11 + self.param_mn)
        self._test_const_expr(self.param_n + self.const_n)
        self._test_const_expr(self.param + self.const)


    def test_diag_vec(self):
        self._test_expr(cvx.diag(self.var_n1))
        self._test_expr(cvx.diag(-self.var_n1))
        self._test_expr(cvx.diag(self.var_1n))
        self._test_expr(cvx.diag(self.var_n))
        self._test_const_expr(cvx.diag(self.param_n1))
        self._test_const_expr(cvx.diag(-self.param_n1))
        self._test_const_expr(cvx.diag(self.param_1n))
        self._test_const_expr(cvx.diag(self.param_n))


    def test_diag_mat(self):
        self._test_expr(cvx.diag(self.var_nn))
        self._test_expr(cvx.diag(cvx.diag(self.var_n1)))
        self._test_expr(cvx.diag(self.var_nn - cvx.diag(self.var_n1)))
        self._test_const_expr(cvx.diag(self.param_nn))
        self._test_const_expr(cvx.diag(cvx.diag(self.param_n1)))
        self._test_const_expr(cvx.diag(self.param_nn - cvx.diag(self.param_n1)))
    

    def test_diff(self):
        self._test_expr(cvx.diff(self.var_n))


    def test_div(self):
        self._test_expr(cvx.sum(self.var_11 / self.const_11))
        self._test_expr(cvx.sum(self.var_mn / self.const_11))
        self._test_expr(cvx.sum(self.var_11 / self.param_11))
        self._test_expr(cvx.sum(self.var_mn / self.param_11))
    

    def test_hstack(self):
        self._test_expr(cvx.hstack([self.var_n1, self.var_np, self.var_nn]))
        self._test_expr(cvx.hstack([self.var_n1, self.const_np, self.param_nn]))
        self._test_const_expr(cvx.hstack([self.param_nn, self.param_n1]))
        self._test_const_expr(cvx.hstack([self.param_np, self.const_n1, self.param_nn]))
    

    def test_index(self):
        self._test_expr(self.var_mn[2:8:2,2:4])
        self._test_expr(self.var_n1[2:7:2])
        self._test_expr(self.var_n1[2:4])
        self._test_expr(self.var_n[1:3])
        self._test_const_expr(self.param_mn[0:8:2, 1:17:3])
        self._test_const_expr(self.param_n1[0:8:2])
        self._test_const_expr(self.param_1n[:5])
        self._test_const_expr(self.param_n[5:])
        self._test_const_expr(self.param_n[:4])
    

    def test_mul(self):
        self._test_expr(self.const_mn * self.var_n1)
        self._test_expr(self.const_mn * self.var_np)
        self._test_expr(self.param_mn * self.var_np)
        self._test_expr(self.const_mn * self.var_n)
        self._test_expr(self.const_n  * self.var_mn.T)
        self._test_expr(self.const_n  * self.var_n)
        self._test_const_expr(self.param_mn * self.param_np)
        self._test_const_expr(self.param_mn * self.const_np)
        self._test_const_expr(self.param_mn * self.param_n1)
        self._test_const_expr(self.param_mn * self.param_n)
        self._test_const_expr(self.const_n  * self.param_mn.T)
        self._test_const_expr(self.const_n  * self.param_n)


    def test_rmul(self):
        self._test_expr(self.var_mn * self.param_np)
        self._test_expr(self.var_mn * self.const_np)
        self._test_expr(self.var_mn * self.param_np)
        self._test_expr(self.var_mn * self.param_np)
        self._test_expr(self.var_n  * self.const_mn.T)
        self._test_expr(self.var_n  * self.const_n)


    def test_multiply(self):
        self._test_expr(cvx.multiply(self.const_mn, self.var_mn))
        self._test_expr(cvx.multiply(self.param_mn, self.var_mn))
        self._test_expr(self.const_n1 * self.var_11)
        self._test_expr(self.const_np * self.var_11)
        self._test_expr(self.param_mn * self.var_11)
        self._test_expr(cvx.multiply(self.var_mn, self.const_mn))
        self._test_expr(self.var_11 * self.const_np)
        self._test_expr(self.var_mn * self.const_11)
        self._test_expr(self.var_11 * self.param_np)
        self._test_expr(self.var_mn * self.param_11)
        self._test_const_expr(cvx.multiply(self.param_mn, self.const_mn))
        self._test_const_expr(cvx.multiply(self.param_mn, self.param_mn))
        self._test_const_expr(cvx.multiply(self.param_n1, self.param_n1))
        self._test_const_expr(cvx.multiply(self.param_n1, self.const_n1))
        self._test_const_expr(cvx.multiply(self.param_11, self.param_11))
        self._test_const_expr(cvx.multiply(self.param_11, self.const_11))
        self._test_const_expr(self.param_mn * self.param_11)
        self._test_const_expr(self.param_11 * self.param_mn)
        self._test_const_expr(self.const_mn * self.param_11)
        self._test_const_expr(self.const_11 * self.param_mn)


    def test_neg(self):
        self._test_expr(-self.var_mn)
        self._test_expr(self.param_mn - self.var_mn)
        self._test_expr(self.const_mn - self.var_mn)
        self._test_const_expr(-self.param_mn)
        self._test_const_expr(-self.param_n1)


    def test_sum(self):
        self._test_expr(cvx.sum(self.var_mn))
        self._test_expr(cvx.sum(self.var_mn + self.var_mn))
        self._test_expr(cvx.sum(self.var_mn + self.const_mn))
        self._test_expr(cvx.sum(self.var_mn + self.param_mn))
        self._test_expr(self.var_11 + cvx.sum(self.const_mn))
        self._test_expr(self.param_11 + cvx.sum(self.var_mn))


    def test_reshape(self):
        self._test_expr(cvx.reshape(self.var_mn, (self.n, self.m)))
        self._test_expr(cvx.reshape(
                self.var_mn + self.const_mn, (self.n, self.m))[1:3, 4:5])
        self._test_expr(cvx.reshape(self.var_mn.T, (self.n, self.m)))
        self._test_expr(cvx.reshape(self.var_mn, (self.n, self.m)).T)
        self._test_expr(cvx.reshape(self.var_mn, (self.n, self.m)) + self.var_mn.T)
        self._test_expr(cvx.reshape(self.var_n1, (1, self.n)) + self.const_n1.T)
        self._test_const_expr(cvx.reshape(self.param_mn, (self.n, self.m)))
        self._test_const_expr(cvx.reshape(self.param_mn + self.const_mn,
                (self.n, self.m)))
        self._test_const_expr(cvx.reshape(self.param_n1, (1, self.n)))
        self._test_const_expr(cvx.reshape(self.param_n1, (self.n,)))
        self._test_const_expr(cvx.reshape(self.param_n1.T, (self.n,)))
        self._test_const_expr(cvx.reshape(self.param_n, (1, self.n)))
        self._test_const_expr(cvx.reshape(self.param_n, (self.n, 1)))
        self._test_const_expr(cvx.reshape(self.param, (1, 1)))
        self._test_const_expr(cvx.reshape(self.param_11, (1,)))
        self._test_const_expr(cvx.reshape(self.param_11, ()))


    def test_trace(self):
        self._test_expr(cvx.trace(self.var_nn))
        self._test_expr(cvx.trace(self.var_11))
        self._test_const_expr(cvx.trace(self.param_nn))
        self._test_const_expr(cvx.trace(cvx.diag(self.param_n1)))
        self._test_const_expr(cvx.trace(self.param_11))
    

    def test_transpose(self):
        self._test_expr(self.var_mn.T[1:3, 2:5])
        self._test_expr((self.var_mn.T + self.const_mn.T)[1:3, 2:5])
        self._test_expr(self.var_n1.T[0, 1:3])
        self._test_expr((self.var_n1 + self.const_n1).T[0, 1:3])
        self._test_expr(self.var_n.T[1:3])
        self._test_const_expr(cvx.sum(self.param_mn.T[1:3, 2:5]))
    

    def test_upper_tri(self):
        self._test_expr(self.var_nn)
        self._test_expr(self.var_nn + self.const_nn)
        self._test_expr(self.var_nn + self.param_nn)


    def test_vstack(self):
        self._test_expr(cvx.vstack(
                [self.var_1n, self.var_mn, self.var_nn])[:,0])
        self._test_expr(cvx.vstack(
                [self.var_1n, self.var_mn])[:,0])
        self._test_expr(cvx.vstack(
                [self.var_1n, self.const_mn, self.param_nn])[:,0])
        self._test_const_expr(cvx.vstack(
                [self.param_n1, self.param_11])[:,0])
        self._test_const_expr(cvx.vstack(
                [self.param_n1, self.param_11, self.const_11])[:,0])
        self._test_const_expr(cvx.vstack(
                [self.param_mn, self.param_pn, self.const_1n])[:,0])



        


if __name__ == '__main__':
   unittest.main()
