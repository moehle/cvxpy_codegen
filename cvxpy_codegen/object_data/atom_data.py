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
import scipy.sparse as sp
from cvxpy_codegen.object_data.coeff_data import CoeffData
from cvxpy_codegen.utils.utils import Counter

LINOP_COUNT = Counter()




class AtomData(ExprData):
    def __init__(self, expr, arg_data):
        ExprData.__init__(self, expr, arg_data)
        self.name = 'linop%d' % LINOP_COUNT.get_count()
        self.data = expr.get_data() # TODO rm? can use self.expr.data instead..
        self.args = arg_data
        self.coeffs = dict()
        self.expr = expr
        self.var_ids = set().union(*[a.var_ids for a in arg_data])
        self.has_offset = any([a.has_offset for a in arg_data])



    # Get the coefficient for each variable.
    def _get_coeffs(self):
        return NotImplemented


    # Get the expression for the offset vector.
    def _get_offset_expr(self):
        return NotImplemented


    def pop_coeffs(self, var_ids):
        coeffs = []
        for vid in var_ids:
            coeffs += [self.coeffs.pop(vid)]
        return coeffs
        

    def get_data(self):
        return self.data


    def force_copy(self):
        for c in self.coeffs:
            c.force_copy()


    def get_matrix(self, x_length, var_offsets):
        coeff_height = self.shape[0] * self.shape[1]
        mat = sp.csc_matrix((coeff_height, x_length), dtype=bool)
        return mat
