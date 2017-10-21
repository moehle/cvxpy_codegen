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
import cvxpy.settings as s
from cvxpy.problems.solvers.ecos_intf import ECOS
import cvxpy
from cvxpy.problems.problem_data.sym_data import SymData
from cvxpy_codegen import codegen
from cvxpy.constraints import NonPos, SOC, Zero

ECOS = ECOS()

HARNESS_C = 'tests/ecos_intf/harness.c.jinja'
CODEGEN_H = 'tests/ecos_intf/codegen.h.jinja'
target_dir = tu.TARGET_DIR



class TestEcosIntf(tu.CodegenTestCase):

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
        self.const_mn = randn(m, n)
        self.const_np = randn(n, p)
        self.const_mp = randn(m, p)
        self.const_pn = randn(p, n)
        self.const_nn = randn(n, n)
        self.const_n1 = randn(n, 1)
        self.const_m1 = randn(m, 1)
        self.const_1n = randn(1, n)
        self.const_11 = randn(1, 1)
        self.const_n  = randn(n,)
        self.const    = randn()



    def test_nonpos(self):
        self._test_constrs([NonPos(-self.var_n1)])
        self._test_constrs([NonPos(self.var_n)])
        self._test_constrs([NonPos(self.var_n1 + self.const_n1)])


    def test_zero(self):
        self._test_constrs([Zero(-self.var_n1)])
        self._test_constrs([Zero(self.var_n)])
        self._test_constrs([Zero(self.var_n1 + self.const_n1)])


    def test_soc(self):
        self._test_constrs([SOC(cvx.sum(self.var_n), -self.var_n)])
        self._test_constrs([SOC(cvx.sum(self.var_n), self.const_n+self.var_n)])
        self._test_constrs([SOC(self.var_n, self.var_mn)])
        self._test_constrs([SOC(self.var_n, self.param_mn)])
        self._test_constrs([SOC(self.param_n, self.var_mn + self.const_mn)])
        self._test_constrs([SOC(self.param_n, self.var_mn + self.param_mn)])
        self._test_constrs([SOC(self.var-self.const, -self.var_n)])




if __name__ == '__main__':
   unittest.main()
