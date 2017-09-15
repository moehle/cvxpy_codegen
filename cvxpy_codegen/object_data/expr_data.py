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

import numpy as np
import scipy.sparse as sp
from cvxpy.lin_ops.lin_op import LinOp # TODO what is this?


class ExprData():

    def __init__(self, expr, arg_data=[], sparsity=None):
        self.args = arg_data
        self.ndims = len(expr.shape)
        if self.ndims == 2:
            self.shape = expr.shape
        elif self.ndims == 1:
            self.shape = (expr.shape[0], 1)
        elif self.ndims == 0:
            self.shape = (1,1)
        else:
            raise Exception("Code generation only supports arrays"
                            "with two or fewer dimensions.")
        if sparsity == None:
            sparsity = sp.csr_matrix(np.full(self.shape, True, dtype=bool))
        self.sparsity = sparsity
        self.length = self.shape[0] * self.shape[1]

    def is_scalar(self):
        return self.ndims == 0

    def is_vector(self):
        return self.ndims == 1

    def is_matrix(self):
        return self.ndims == 2

