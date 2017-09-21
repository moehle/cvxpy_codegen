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
        m = 3
        n = 5
        p = 4
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
        self.var_n  = cvx.Variable((n,  ), name='var_n' )
        self.var    = cvx.Variable((    ), name='var'   )
        self.param_mn = cvx.Parameter((m, n), name='param_mn', value=np.random.randn(m, n))
        self.param_np = cvx.Parameter((n, p), name='param_np', value=np.random.randn(n, p))
        self.param_mp = cvx.Parameter((m, p), name='param_mp', value=np.random.randn(m, p))
        self.param_pn = cvx.Parameter((p, n), name='param_pn', value=np.random.randn(p, n))
        self.param_nn = cvx.Parameter((n, n), name='param_nn', value=np.random.randn(n, n))
        self.param_n1 = cvx.Parameter((n, 1), name='param_n1', value=np.random.randn(n, 1))
        self.param_1n = cvx.Parameter((1, n), name='param_1n', value=np.random.randn(1, n))
        self.param_11 = cvx.Parameter((1, 1), name='param_11', value=np.random.randn(1, 1))
        self.param_n  = cvx.Parameter((n,  ), name='param_n',  value=np.random.randn(n))
        self.param    = cvx.Parameter((    ), name='param',    value=np.random.randn())
        self.const_mn = np.random.randn(m, n)
        self.const_np = np.random.randn(n, p)
        self.const_mp = np.random.randn(m, p)
        self.const_pn = np.random.randn(p, n)
        self.const_nn = np.random.randn(n, n)
        self.const_n1 = np.random.randn(n, 1)
        self.const_m1 = np.random.randn(m, 1)
        self.const_1n = np.random.randn(1, n)
        self.const_11 = np.random.randn(1, 1)
        self.const_n  = np.random.randn(n,  )
        self.const    = np.random.randn(    )



    def test_nonpos(self):
        self._test_constrs([NonPos(-self.var_n1)])
        self._test_constrs([NonPos(self.var_n)])
        self._test_constrs([NonPos(self.var_n1 + self.const_n1)])


    def test_nonpos(self):
        self._test_constrs([Zero(-self.var_n1)])
        self._test_constrs([Zero(self.var_n)])
        self._test_constrs([Zero(self.var_n1 + self.const_n1)])


    def test_soc(self):
        self._test_constrs([SOC(cvx.sum(self.var_n), -self.var_n)])
        self._test_constrs([SOC(cvx.sum(self.var_n), self.const_n+self.var_n)])
        self._test_constrs([SOC(self.var_n, self.var_mn)])
        self._test_constrs([SOC(self.param_n, self.var_mn + self.param_mn)])
        self._test_constrs([SOC(self.var-self.const, -self.var_n)])





    def _test_constrs(self, constrs, printing=False):
        prob = cvx.Problem(cvxpy.Minimize(0), constrs)
        prob_data = prob.get_problem_data("ECOS")
        data = prob_data[0]
        inverses = prob_data[2]
        for inv in inverses:
            if hasattr(inv, 'r'):
                true_obj_offset = inv.r
        true_obj_coeff   = data[s.C]
        true_obj_offset += data[s.OFFSET]
        true_eq_coeff    = data[s.A]
        true_eq_offset   = data[s.B]
        true_leq_coeff   = data[s.G]
        true_leq_offset  = data[s.H]

        # Do code generation
        template_vars = codegen(prob, target_dir, dump=True, include_solver=False)

        # Set up test harness.
        render(target_dir, template_vars, HARNESS_C, 'harness.c')
        render(target_dir, template_vars, CODEGEN_H, 'codegen.h')
        test_data = self._run_test(target_dir)
        test_obj_coeff  = np.array(test_data['obj_coeff'])
        test_obj_offset = np.array(test_data['obj_offset'])
        test_eq_coeff  = sp.csc_matrix((test_data['eq_nzval'],
                                        test_data['eq_rowidx'],
                                        test_data['eq_colptr']),
                                        shape = (test_data['eq_shape0'],
                                                 test_data['eq_shape1']))
        test_eq_offset = np.array(test_data['eq_offset'])
        test_leq_coeff = sp.csc_matrix((test_data['leq_nzval'],
                                        test_data['leq_rowidx'],
                                        test_data['leq_colptr']),
                                        shape = (test_data['leq_shape0'],
                                                 test_data['leq_shape1']))
        test_leq_offset = np.array(test_data['leq_offset'])

        if printing:
            print '\nTest objective coeff:\n',   test_obj_coeff
            print '\nTrue objective coeff:\n',   true_obj_coeff

            print '\nTest objective offset:\n',  test_obj_offset
            print '\nTrue objective offset:\n',  true_obj_offset

            print '\nTest equality coeff:\n',    test_eq_coeff
            print '\nTrue equality coeff:\n',    true_eq_coeff

            print '\nTest equality offset:\n',   test_eq_offset
            print '\nTrue equality offset:\n',   true_eq_offset

            print '\nTest inequality coeff:\n',  test_leq_coeff
            print '\nTrue inequality coeff:\n',  true_leq_coeff

            print '\nTest inequality offset:\n', test_leq_offset
            print '\nTrue inequality offset:\n', true_leq_offset

        if not true_obj_coeff is None:
            self.assertAlmostEqualMatrices(true_obj_coeff,  test_obj_coeff)
        if not true_obj_offset is None:
            self.assertAlmostEqualMatrices(true_obj_offset, test_obj_offset)
        if not true_eq_coeff is None:
            self.assertAlmostEqualMatrices(true_eq_coeff,   test_eq_coeff)
        if not test_eq_offset is None:
            self.assertAlmostEqualMatrices(true_eq_offset,  test_eq_offset)
        if not test_leq_coeff is None:
            self.assertAlmostEqualMatrices(true_leq_coeff,  test_leq_coeff)
        if not test_leq_offset is None:
            self.assertAlmostEqualMatrices(true_leq_offset, test_leq_offset)



    def _run_test(self, target_dir):
        prev_path = os.getcwd()
        os.chdir(target_dir)
        output = subprocess.check_output(
                     ['gcc', 'harness.c', 'expr_handler.c', 'solver_intf.c',
                      '-g', '-o' 'main'],
                     stderr=subprocess.STDOUT)
        exec_output = subprocess.check_output(['./main'], stderr=subprocess.STDOUT)
        os.chdir(prev_path)
        return json.loads(exec_output.decode("utf-8"))
        


if __name__ == '__main__':
   unittest.main()
