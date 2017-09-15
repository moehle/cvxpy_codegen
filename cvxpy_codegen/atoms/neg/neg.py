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

from cvxpy_codegen.object_data.atom_data import AtomData
from cvxpy_codegen.object_data.linop_coeff_data import LinOpCoeffData

def atomdata_neg(expr, arg_data, arg_pos):
    return AtomData(expr, arg_data,
                   macro_name = "neg",
                   sparsity = arg_data[0].sparsity,
                   inplace = True)


def coeffdata_neg(linop, args, var):
    return LinOpCoeffData(linop, args, var,
                          sparsity = args[0].sparsity,
                          inplace = True,
                          macro_name = 'neg_coeffs')
