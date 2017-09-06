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

from cvxpy_codegen.object_data.expr_data import ExprData
from cvxpy_codegen.utils.utils import Counter
import scipy.sparse as sp
from cvxpy.lin_ops.lin_op import LinOp

CONST_COUNT = Counter()
CONST_ID = "CONST_ID"


class ConstData(ExprData):
    def __init__(self, expr):
        value = expr.data if isinstance(expr, LinOp) else expr.value
        sparsity = sp.csr_matrix(value, dtype=bool)
        ExprData.__init__(self, expr, sparsity=sparsity)
        self.type = 'const'
        self.name = 'const%d' % CONST_COUNT.get_count()
        self.value = sp.csr_matrix(value)
        self.rowptr = self.value.indptr
        self.colidx = self.value.indices
        self.nzval = self.value.data
        self.nnz = self.value.nnz
        self.var_ids = [CONST_ID]
        self.mem_name = self.name
        self.cname = self.storage.name

    @property
    def storage(self):
        return self
