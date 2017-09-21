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
from cvxpy_codegen.object_data.const_data import CONST_ID
import scipy.sparse as sp
from cvxpy_codegen.object_data.coeff_data import CoeffData
from cvxpy_codegen.object_data.atom_data import AtomData




class BilinAtomData(AtomData, object):
    def __init__(self, expr, arg_data):
        super(BilinAtomData, self).__init__(expr, arg_data)

        self.has_offset = False
        if self.args[0].var_ids:
            self.const_arg = 1
            self.var_arg = 0
        elif self.args[1].var_ids:
            self.const_arg = 0
            self.var_arg = 1
        else:
            self.var_arg = None
            self.has_offset = True
             
        if self.var_arg:
            if self.args[self.var_arg].has_offset:
                self.has_offset = True


    # Get the coefficient for each variable.
    def get_coeffs(self):
        if not self.var_arg is None:
            coeff_args = [None, None]
            for vid in self.args[self.var_arg].var_ids:
                coeff_args[self.const_arg] = self.args[self.const_arg]
                if isinstance(self.args[self.var_arg], AtomData):
                    coeff_args[self.var_arg] = self.args[self.var_arg].coeffs[vid]
                else:
                    coeff_args[self.var_arg] = self.args[self.var_arg]
                coeff = self.get_coeff_data(coeff_args, vid) 
                self.coeffs.update({vid : coeff})
        return self.coeffs


    # Get the expression for the offset vector.
    def get_offset_expr(self):
        if not self.var_arg:
            super(BilinAtomData, self).get_offset_expr()
        elif self.has_offset:
            offset_args = [None, None]
            offset_args[self.const_arg] = self.args[self.const_arg]
            offset_args[self.var_arg] = self.args[self.var_arg].offset_expr
            self.offset_expr = self.get_atom_data(self.expr, offset_args)
        return self.offset_expr


    def force_copy(self):
        for c in self.coeffs:
            c.force_copy()


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
