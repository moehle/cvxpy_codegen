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




class AffAtomData(AtomData, object):
    def __init__(self, expr, arg_data):
        super(AffAtomData, self).__init__(expr, arg_data)
        self._get_coeffs()
        self._get_offset_expr()


    # Get the coefficient for each variable.
    def _get_coeffs(self):
        for vid in self.var_ids:
            coeff_args = []
            for arg in self.args:
                if vid in arg.var_ids:
                    coeff_args += [arg.coeffs[vid]]
                    #if isinstance(arg, AtomData):
                    #    coeff_args += [arg.coeffs[vid]]
                    #else:
                    #    coeff_args += [arg]
            coeff = self.get_coeff_data(coeff_args, vid)
            self.coeffs.update({vid : coeff})


    # Get the expression for the offset vector.
    def _get_offset_expr(self):
        arg_count = 0
        arg_pos = [] # Store the argument positions.
        print
        print type(self)
        print self.has_offset
        print self.args
        print self.args[0].has_offset
        if self.has_offset:
            offset_args = []
            for arg in self.args:
                if arg.has_offset:
                    arg_pos += [arg_count]
                    offset_args += [arg.offset_expr]
                    #arg_pos += [arg_count]
                    #if isinstance(arg, AtomData):
                    #    offset_args += [arg.offset_expr]
                    #else:
                    #    offset_args += [arg]
                arg_count += 1
            self.offset_expr = self.get_atom_data(self.expr, offset_args, arg_pos)
        raise Exception


    def get_data(self):
        return self.data


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
