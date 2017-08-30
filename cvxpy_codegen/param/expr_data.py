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
from cvxpy_codegen.utils.utils import Counter, spzeros
from cvxpy.lin_ops.lin_op import LinOp


EXPR_COUNT = Counter()
CONST_COUNT = Counter()
LINOP_COUNT = Counter()
CONST_ID = "CONST_ID"


class ExprData():

    def __init__(self, expr, arg_data=[], sparsity=None):
        if sparsity == None:
            sparsity = sp.csr_matrix(np.full(expr.size, True, dtype=bool))
        self.sparsity = sparsity
        self.args = arg_data
        self.size = expr.size
        self.length = expr.size[0] * expr.size[1]

    def is_vector(self):
        return (expr.size[0] == 1) or (expr.size[1] == 1)

    def is_scalar(self):
        return expr.size == (1,1)



class ParamData(ExprData):
    def __init__(self, expr):
        ExprData.__init__(self, expr)
        self.type = 'param'
        self.name = expr.name()
        if expr.value is None:
            self.value = np.squeeze(np.random.randn(*expr.size))
        else:
            self.value = expr.value
        self.var_ids = [CONST_ID]
        self.mem_name = self.name
        self.is_scalar = True if self.size == (1,1) else False
        self.is_column = True if self.size[1] == 1 else False
        self.is_row    = True if self.size[0] == 1 else False # TODO add tests for these
        
    @property
    def storage(self):
        return self



class CbParamData(ExprData):
    def __init__(self, expr, arg_data):
        sparsity = arg_data[0].sparsity
        ExprData.__init__(self, expr, arg_data=arg_data, sparsity=sparsity)
        self.type = 'cbparam'
        self.name = expr.name()
        self.cbp_name = expr.name()
        self.var_ids = [CONST_ID]
        self.inplace = True
        self.mem_name = arg_data[0].name

    @property
    def storage(self):
        return self.args[0].storage




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

    @property
    def storage(self):
        return self



class AtomData(ExprData):
    def __init__(self, expr, arg_data, 
                 sparsity=None, inplace=False, macro_name=None,
                 work_int=0, work_float=0, size=None, data=None,
                 copy_arg=0):
        ExprData.__init__(self, expr, arg_data=arg_data, sparsity=sparsity)
        self.type = 'expr'
        self.name = 'expr%d' % EXPR_COUNT.get_count()
        if size is not None:
           self.size = size
        self.macro_name = macro_name
        self.inplace = inplace
        self.copy_arg = copy_arg # If self.make_copy is True, copy this argument.
        self.work_int = work_int
        self.work_float = work_float
        has_const_or_param = \
                any([a.type =='const' or a.type =='param' for a in arg_data])
        if inplace and has_const_or_param:
            self.make_copy = True
        else:
            self.make_copy = False
        self.data = data

    @property
    def storage(self):
        if self.inplace and not self.make_copy:
            return self.args[self.copy_arg].storage
        else:
            return self

    def force_copy(self):
        self.make_copy = True
