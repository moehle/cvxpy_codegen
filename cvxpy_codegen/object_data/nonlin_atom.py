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

from cvxpy_codegen.object_data.atom_data import AtomData




class NonLinAtomData(AtomData):
    def __init__(self, expr, arg_data):
        super(NonLinAtomData, self).__init__(expr, arg_data)
        self._get_coeffs()
        self._get_offset_expr()



    # Get the coefficient for each variable.
    def _get_coeffs(self):
        pass


    # Get the expression for the offset vector.
    def _get_offset_expr(self):
        offset_exprs = []
        for arg in self.args:
            offset_exprs += [arg.offset]
            #if arg.has_offset: # TODO make all this simpler
            #    if isinstance(arg, AtomData):
            #        offset_exprs += [arg.offset_expr]
            #    else:
            #        offset_exprs += [arg]
        self.offset_expr = self.get_atom_data(self.expr, offset_exprs)
        return self.offset_expr


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

