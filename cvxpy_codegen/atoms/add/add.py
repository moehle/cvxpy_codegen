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

def atomdata_add(expr, arg_data):

    if len(arg_data) == 1:
        return AtomData(expr, arg_data,
                        inplace = True,
                        sparsity = arg_data[0].sparsity,
                        macro_name = 'null')

    else:
        if any([a.sparsity.shape==(1,1) for a in arg_data]):
            sparsity = None
            work_int    = arg_data[0].shape[1]
            work_float  = arg_data[0].shape[1]
        else:
            sparsity = sum([a.sparsity for a in arg_data])
            work_int    = arg_data[0].shape[1]
            work_float  = arg_data[0].shape[1]
        work_varargs    = len(arg_data) # This is a varargs atom.
        return AtomData(expr, arg_data,
                        sparsity = sparsity,
                        work_int = work_int,
                        work_float = work_float,
                        work_varargs = work_varargs,
                        macro_name = 'add')


def coeffdata_add(linop, args, var):
    if len(args) == 1:
        return LinOpCoeffData(linop, args, var,
                              inplace = True,
                              sparsity = args[0].sparsity,
                              macro_name = 'null')

    else:
        work_coeffs = len(args) # This is a varargs linop.
        sparsity = sum([a.sparsity for a in args])
        work_int    = sparsity.shape[1]
        work_float  = sparsity.shape[1]
        return LinOpCoeffData(linop, args, var,
                              sparsity = sparsity,
                              work_int = work_int,
                              work_float = work_float,
                              work_coeffs = work_coeffs,
                              macro_name = 'add_coeffs')
