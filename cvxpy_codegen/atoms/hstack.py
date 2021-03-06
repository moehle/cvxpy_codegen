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
import scipy.sparse as sp

def getdata_hstack(expr, arg_data):
    data_list = []
    sparsity = arg_data[0].sparsity
    data = arg_data[0]
    for i, arg in enumerate(arg_data[1:]):
        sparsity = sp.hstack([sparsity, arg.sparsity])
        data = AtomData(expr, [data, arg],
                               macro_name = "hstack",
                               sparsity = sparsity,
                               size = sparsity.shape)
        data_list += [data]

    return data_list
