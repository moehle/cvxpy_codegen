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

#from cvxpy_codegen.atoms import get_atom_data, get_coeff_data
from cvxpy_codegen.object_data.expr_data import ExprData
from cvxpy_codegen.object_data.const_data import CONST_ID
from cvxpy_codegen.utils.utils import spzeros # TODO rm
import scipy.sparse as sp
from cvxpy_codegen.object_data.coeff_data import CoeffData
from cvxpy_codegen.utils.utils import Counter

LINOP_COUNT = Counter()




class AtomData(ExprData):
    def __init__(self, expr, arg_data):
        ExprData.__init__(self, expr, arg_data)
        self.type = 'linop'
        self.opname = type(expr)
        self.name = 'linop%d' % LINOP_COUNT.get_count()
        self.data = expr.get_data()
        self.args = arg_data
        self.coeffs = dict()
        self.expr = expr
        self.var_ids = set().union(*[a.var_ids for a in arg_data])
        if any([a.has_offset for a in arg_data]):
            self.has_offset = True 
        else:
            self.has_offset = False


    # Get the coefficient for each variable.
    def get_coeffs(self):
        return dict()


    # Get the expression for the offset vector.
    def get_offset_expr(self):
        arg_pos = range(len(self.args)) # args are in order.
        offset_exprs = []
        for arg in self.args:
            if arg.has_offset: # TODO make all this simpler
                if isinstance(arg, AtomData):
                    offset_exprs += [arg.offset_expr]
                else:
                    offset_exprs += [arg]
        self.offset_expr = self.get_atom_data(self.expr, offset_exprs, arg_pos)
        return self.offset_expr


    def get_data(self):
        return self.data



    def force_copy(self):
        for c in self.coeffs:
            c.force_copy()


    def get_matrix(self, x_length, var_offsets):
        coeff_height = self.shape[0] * self.shape[1]
        mat = sp.csc_matrix((coeff_height, x_length), dtype=bool)
        return mat
