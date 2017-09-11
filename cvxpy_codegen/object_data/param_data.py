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
import numpy as np



class ParamData(ExprData):
    def __init__(self, expr):
        ExprData.__init__(self, expr)
        self.type = 'param'
        self.name = expr.name()
        if expr.value is None:
            self.value = np.squeeze(np.random.randn(*expr.shape))
        else:
            self.value = expr.value
        self.var_ids = []
        self.mem_name = self.name
        self.is_scalar = True if self.shape == (1,1) else False
        self.is_column = True if self.shape[1] == 1 else False
        self.is_row    = True if self.shape[0] == 1 else False # TODO add tests for these
        self.cname = self.name
        self.has_offset = True
        
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
        self.var_ids = []
        self.inplace = True
        self.mem_name = arg_data[0].name
        self.cname = self.storage.name
        self.has_offset = True

    @property
    def storage(self):
        return self.args[0].storage
