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

def getdata_add(expr, arg_data):
    data = arg_data[0]
    sparsity = arg_data[0].sparsity

    data_list = []
    for i, arg in enumerate(arg_data[1:]):
        if data.size == (1,1):
            sparsity = arg.sparsity
            macro_name = 'scalar_add'
            work_int = 0
            work_float = 0
            inplace = True
            copy_arg = 1
        elif arg.size == (1,1):
            sparsity = data.sparsity
            macro_name = 'scalar_radd'
            work_int = 0
            work_float = 0
            inplace = True
            copy_arg = 0
        else:
            sparsity += arg.sparsity
            macro_name = 'add'
            work_int = arg.sparsity.shape[1]
            work_float = arg.sparsity.shape[1]
            inplace = False
            copy_arg = 0
        data = AtomData(expr, arg_data = [data, arg],
                        macro_name = macro_name,
                        sparsity = sparsity,
                        work_int = work_int,
                        work_float = work_float,
                        inplace = inplace,
                        copy_arg = copy_arg)
        data_list += [data]

    return data_list
