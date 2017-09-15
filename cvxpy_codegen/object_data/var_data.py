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
import scipy.sparse as sp
from cvxpy_codegen.object_data.const_data import CONST_ID
import cvxpy.settings as s

VAR_NAMES = []


class VarData(ExprData):
    def __init__(self, expr):
        ExprData.__init__(self, expr)
        self.type = 'var'
        self.id = expr.id
        #self.name = 'var%d' % self.id
        self.name = expr.name()
        self.arg_data = []
        self.sparsity = sp.csr_matrix(sp.eye(self.length, dtype=bool))
        self.var_ids = {self.id}
        self.storage = self # Where is the coefficient stored in C?
        self.has_offset = False
        self.coeffs = {self.id : self}
        self.vid = self.id

        # TODO option to not ignore default names.
        self.is_named = False
        if self.name in VAR_NAMES:
            raise TypeError('Duplicate variable name %s' % self.name)
        if not self.name == "%s%d" % (s.VAR_PREFIX, self.id):
            self.is_named = True


    
    # TODO Combine with LinopData
    def get_matrix(self, x_length, var_offsets):
        coeff_height = self.shape[0] * self.shape[1]
        mat = sp.csc_matrix((coeff_height, x_length), dtype=bool)
        for vid, coeff in self.coeffs.items():
            if not (vid == CONST_ID):
                start = var_offsets[vid]
                coeff_width = coeff.sparsity.shape[1]
                pad_left = start
                pad_right = x_length - coeff_width - pad_left
                mat += sp.hstack([sp.csc_matrix((coeff_height, pad_left), dtype=bool),
                                  coeff.sparsity,
                                  sp.csc_matrix((coeff_height, pad_right), dtype=bool)])
        return mat
