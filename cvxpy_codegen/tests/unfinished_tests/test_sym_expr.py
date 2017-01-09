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
import numpy
import cvxpy_codegen.linop_sym.sym_matrix as sym
import scipy.sparse as sp
 
m = 5
n = 6
p = 8
dens = .3
A = sp.rand(m,n, dens)
B = sp.rand(n,p, dens)


#class TestSymMatrix(unittest.TestCase):
#
#    #def test_symbolic_equality(self):
#    #    As = sym.as_sym_matrix(A)
#    #    Bs = sym.as_sym_matrix(B)
#    #    self.assertTrue(As == As)
#    #    self.assertTrue(Bs == Bs)
#    #    self.assertFalse(As == Bs)
#
#    #def test_symbolic_matrix_matrix_multiply(self):
#    #    As = sym.as_sym_matrix(A)
#    #    Bs = sym.as_sym_matrix(B)
#    #    C1 = As*Bs
#    #    C2 = sym.as_sym_matrix(A*B)
#    #    self.assertTrue(C1 == C2)
#
#    def test_value(self):
#        n1 = 5
#        n2 = 3
#        val3 = 
#
#        a
#


        
    



if __name__ == '__main__':
    unittest.main()
