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


TARGET_DIR = os.path.join(os.getcwd(), 'cg_build')


class CodegenTestCase(unittest.TestCase):
    EPS = 1e-5


    def assertAlmostEqualMatrices(self, A, B, eps=None):
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
            sys.stdout.buffer.write(output)
        return exit_code


    def _run_codegen_test(self, prob, module, class_name, method_name):
        codegen(prob, TARGET_DIR)
        self.install_custom_solver(TARGET_DIR)
        exit_code = self._run_isolated_test(module, class_name, method_name)
        self.assertEqual(exit_code, 0)
