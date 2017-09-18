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
EXPR_COUNT = Counter()


class ConstExprData(ExprData):
    def __init__(self, expr, arg_data, 
                 sparsity=None, inplace=False, macro_name=None,
                 work_int=0, work_float=0, work_varargs=0,
                 shape=None, data=None, copy_arg=0):
        ExprData.__init__(self, expr, arg_data=arg_data, sparsity=sparsity)
        self.type = 'expr'
        self.name = 'expr%d' % EXPR_COUNT.get_count()
        if shape is not None: # Override the shape. # TODO remove?
           self.shape = shape
        self.macro_name = macro_name
        self.inplace = inplace
        self.copy_arg = copy_arg # If self.make_copy is True, copy this argument.
        self.work_int = work_int
        self.work_float = work_float
        self.work_varargs = work_varargs
        has_const_or_param = \
                any([a.type =='const' or a.type =='param' for a in arg_data])
        if inplace and has_const_or_param:
            self.make_copy = True
        else:
            self.make_copy = False
        self.data = data
        self.cname = self.storage.name
        self.var_ids = []
        self.has_offset=True
        self.coeffs = {} # TODO true

    @property
    def storage(self):
        #print(self.macro_name)
        if self.inplace and not self.make_copy:
            return self.args[self.copy_arg].storage
        else:
            return self
        # TODO is copy_arg really necessary?

    def force_copy(self):
        self.make_copy = True
