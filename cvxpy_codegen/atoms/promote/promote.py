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

from cvxpy_codegen.object_data.const_expr_data import ConstExprData
from cvxpy_codegen.object_data.coeff_data import CoeffData
from cvxpy_codegen.object_data.aff_atom_data import AffAtomData
import numpy as np
import scipy.sparse as sp


class PromoteData(AffAtomData):

    def get_atom_data(self, expr, arg_data, arg_pos):
        return ConstExprData(expr, arg_data,
                             macro_name = "promote")


    def get_coeff_data(self, args, var):
        sparsity = sp.csr_matrix(np.ones((self.length,1)) * args[0].sparsity)
        return CoeffData(self, args, var,
                         sparsity = sparsity,
                         macro_name = 'promote_coeffs')
