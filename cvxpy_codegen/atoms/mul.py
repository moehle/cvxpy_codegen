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

from cvxpy_codegen.param.expr_data import AtomData

def getdata_mul(expr, arg_data):
    if arg_data[0].size == (1,1):
        return [AtomData(expr, arg_data,
                         inplace = True,
                         copy_arg = 1,
                         macro_name = "scalar_mul",
                         sparsity = arg_data[1].sparsity)]
    if arg_data[1].size == (1,1):
        return [AtomData(expr, arg_data,
                         inplace = True,
                         copy_arg = 0,
                         macro_name = "scalar_rmul",
                         sparsity = arg_data[0].sparsity)]
    else:
        return [AtomData(expr, arg_data,
                         macro_name = "mul",
                         sparsity = arg_data[0].sparsity * arg_data[1].sparsity,
                         work_int = arg_data[0].sparsity.shape[1],
                         work_float = arg_data[0].sparsity.shape[1])]
