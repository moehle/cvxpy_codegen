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
from cvxpy_codegen.code_generator import CodeGenerator

ECOS = ECOS()

HARNESS_C = 'tests/expr_handler/harness.c.jinja'
CODEGEN_H = 'tests/expr_handler/codegen.h.jinja'
target_dir = tu.TARGET_DIR



class TestLinopHandler(tu.CodegenTestCase):

    # Define some Parameters and Constants to build tests with.
    def setUp(self):
        np.random.seed(0)
        m = 8
        n = 12
        p = 10
        self.m = m
        self.n = n
        self.p = p
        self.var_n1 = cvx.Variable((n, 1), name='var_n1')
        self.const_mn = np.random.randn(m, n)
        self.const_m1 = np.random.randn(m, 1)
        self.ones_n1 = np.ones((n, 1))




    ###########################
    # FUNCTIONS OF PARAMETERS #
    ###########################

    def test(self):
        aff_expr = self.const_mn * self.var_n1 + self.const_m1
        c = NonPos(aff_expr)
        
        constrs = [c]
        objective = self.ones_n1 * x + self.const_11
        _test_prob(obj, constrs, printing=True)






    def _test_prob(self, obj, constrs, printing=False):
        prob = cvx.Problem(cvxpy.Minimize(obj), constrs)

        SC = construct_solving_chain(prob, solver="ECOS")
        for r in SC.reductions:
            if isinstance(r, ECOS_reduction):
                data, inv_data = r.apply(prob)
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
                     ['gcc', 'harness.c', 'expr_handler.c', '-o' 'main'],
                     stderr=subprocess.STDOUT)
        exec_output = subprocess.check_output(['./main'], stderr=subprocess.STDOUT)
        os.chdir(prev_path)
        return json.loads(exec_output.decode("utf-8"))
        


if __name__ == '__main__':
   unittest.main()
