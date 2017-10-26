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

import os
import shutil as sh
import imp
from cvxpy_codegen.utils.utils import PKG_PATH
import unittest
import subprocess
import sys
import numpy as np
import scipy.sparse as sp
from cvxpy_codegen import codegen
import cvxpy as cvx
from cvxpy_codegen.utils.utils import render, make_target_dir
import cvxpy.settings as s
import json
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain

from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS


TARGET_DIR = os.path.join(os.getcwd(), 'cg_build')
HARNESS_C = 'tests/harness.c.jinja'



def convert_to_matrix(A):
    if sp.issparse(A):
        A = A.toarray()
    if len(A.shape)==0:
        A = np.expand_dims(A, 0)
        return np.expand_dims(A, 1)
    if len(A.shape)==1: # Vectors are assumed to be column vectors.
        return np.expand_dims(A, 1)
    if len(A.shape)==2:
        return A
    else:
        RuntimeError("Shape is greater than 2")



class CodegenTestCase(unittest.TestCase):
    EPS = 1e-5




    def assertAlmostEqualMatrices(self, A, B, eps=None):

        if not(A is None):
            if any([s == 0 for s in A.shape]):
                A = None
        if not(B is None):
            if any([s == 0 for s in B.shape]):
                B = None
        if A is None and B is None:
            return
        if A is None or B is None:
            self.assertTrue(False)

        A = convert_to_matrix(A)
        B = convert_to_matrix(B)

        if eps == None:
            eps = self.EPS
        D =  abs(A-B)
        if sp.issparse(D):
            D = D.toarray()
        self.assertTrue(np.all(D <= eps))
        

    def assertEqualMatrices(self, A, B, eps=None):
        self.assertAlmostEqualMatrices(A, B, eps=0)
        

    def assertAlmostEqualLists(self, A, B, eps=None):
        self.assertEqual(len(A), len(B))
        if eps == None:
            eps = self.EPS
        for i in range(len(A)):
            self.assertTrue(abs(A[i] - B[i]) <= eps)


    def assertEqualLists(self, A, B, eps=None):
        self.assertAlmostEqualLists(A, B, eps=0)
        

    def install_custom_solver(self, cg_path):
        prev_path = os.getcwd()
        os.chdir(cg_path)
        os.system('python setup.py --quiet install')
        os.chdir(prev_path)


    def _run_isolated_test(self, module, cls, method):
        command = "%s.%s.%s" % (module, cls, method)
        try:
            output = subprocess.check_output(
                    ['python', '-m', 'unittest', command],
                    stderr=subprocess.STDOUT)
            exit_code = 0
        except subprocess.CalledProcessError as exc:
            output = exc.output
            exit_code = exc.returncode
            sys.stdout.write(output)
        return exit_code


    def _run_codegen_test(self, prob, method_name):
        codegen(prob, TARGET_DIR)
        self.install_custom_solver(TARGET_DIR)
        exit_code = self._run_isolated_test(self.module, self.class_name, method_name)
        self.assertEqual(exit_code, 0)



    def _test_constrs(self, constrs, **kwargs):
        prob = cvx.Problem(cvx.Minimize(0), constrs)
        self._test_prob(prob, **kwargs)


    def _test_expr(self, expr, **kwargs):
        prob = cvx.Problem(cvx.Minimize(cvx.sum(expr)), [])
        self._test_prob(prob, **kwargs)


    def _test_const_expr(self, expr, **kwargs):
        obj = cvx.Minimize(cvx.sum(expr) + cvx.Variable(name='extra_var'))
        prob = cvx.Problem(obj, [])
        self._test_prob(prob, **kwargs)

    def _test_prob(self, prob, printing=False):
        sc = construct_solving_chain(prob, solver="ECOS")

        has_params = False
        for r in sc.reductions:
            if isinstance(r, EvalParams):
                eval_params = r
                has_params = True
            elif isinstance(r, ConeMatrixStuffing):
                cone_matrix_stuffing = r
            elif isinstance(r, Dcp2Cone):
                dcp2cone = r
            elif isinstance(r, CvxAttr2Constr):
                cvx_attr2constr= r
            elif isinstance(r, ECOS):
                ecos = r
            else:
                raise Exception('Unrecognized reduction.')

        prob, inv_data = dcp2cone.apply(prob)
        prob, __ = cvx_attr2constr.apply(prob)
        if has_params:
            prob_cvx, __ = eval_params.apply(prob)
            prob_cvx, inv_matrixstuffing = cone_matrix_stuffing.apply(prob_cvx)
        else:
            prob_cvx, inv_matrixstuffing = cone_matrix_stuffing.apply(prob)
        data, __ = ecos.apply(prob_cvx)

        true_obj_offset = inv_matrixstuffing.r
        true_obj_coeff   = data[s.C]
        true_obj_offset += data[s.OFFSET]
        true_eq_coeff    = data[s.A]
        true_eq_offset   = data[s.B]
        true_leq_coeff   = data[s.G]
        true_leq_offset  = data[s.H]

        if not true_eq_offset is None:
            true_eq_offset = -true_eq_offset
        if not true_leq_offset is None:
            true_leq_offset = -true_leq_offset

        obj = prob.objective
        constraints = prob.constraints
        inv_data = inv_matrixstuffing

        # Do code generation
        template_vars = codegen(prob, TARGET_DIR, dump=True,
                                include_solver=False, solver='ecos')

        # Set up test harness.
        render(TARGET_DIR, template_vars, HARNESS_C, 'harness.c')
        test_data = self._run_test(TARGET_DIR)
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
            print('\nTest objective coeff  :\n',   test_obj_coeff)
            print('\nTrue objective coeff  :\n',   true_obj_coeff)

            print('\nTest objective offset :\n',   test_obj_offset)
            print('\nTrue objective offset :\n',   true_obj_offset)

            print('\nTest equality coeff  :\n',    test_eq_coeff)
            print('\nTrue equality coeff  :\n',    true_eq_coeff)

            print('\nTest equality offset :\n',    test_eq_offset)
            print('\nTrue equality offset :\n',    true_eq_offset)

            print('\nTest inequality coeff  :\n',  test_leq_coeff.todense())
            print('\nTrue inequality coeff  :\n',  true_leq_coeff.todense())

            print('\nTest inequality offset :\n',  test_leq_offset)
            print('\nTrue inequality offset :\n',  true_leq_offset)

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
        os.chdir(TARGET_DIR)
        output = subprocess.check_output(
                     ['gcc', 'harness.c', 'expr_handler.c', 'solver_intf.c',
                      '-g', '-o' 'main'],
                     stderr=subprocess.STDOUT)
        exec_output = subprocess.check_output(['./main'], stderr=subprocess.STDOUT)
        os.chdir(prev_path)
        return json.loads(exec_output.decode("utf-8"))
