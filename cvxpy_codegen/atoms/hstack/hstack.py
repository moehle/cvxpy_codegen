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

from cvxpy_codegen.object_data.const_expr_data import ConstExprData
from cvxpy_codegen.object_data.coeff_data import CoeffData
from cvxpy_codegen.object_data.aff_atom_data import AffAtomData
import scipy.sparse as sp

class HStackData(AffAtomData):

    def get_atom_data(self, expr, arg_data, arg_pos):

        offsets = []
        horz_offset = 0
        for i, a in enumerate(expr.args):
            if i in arg_pos:
                offsets += [horz_offset]
            horz_offset += a.shape[0]

        work_varargs = len(arg_data) # This is a varargs atom.
        work_int = len(arg_data)

        #ndims = len(expr.shape)
        #if ndims == 2:
        #    shape = expr.shape
        #elif ndims == 1:
        #    shape = (expr.shape[0], 1)
        #elif ndims == 0:
        #    shape = (1,1)
        #else:
        #    raise Exception("Code generation only supports arrays"
        #                    "with two or fewer dimensions.")

        sparsity = sp.hstack([a.sparsity for a in arg_data])

        return ConstExprData(expr, arg_data,
                             macro_name = "hstack",
                             sparsity = sparsity,
                             work_varargs = work_varargs,
                             work_int = work_int,
                             data = offsets)



    def get_coeff_data(self, args, var):

        # TODO replace with arg_pos:
        vert_offset = 0
        offsets = []
        for a in self.args:
            if var in a.var_ids:
                offsets += [vert_offset]
            vert_offset += a.length

        var_size = args[0].sparsity.shape[1]
        sparsity = sp.lil_matrix((self.length, var_size))
        for a, os in zip(args, offsets):
            m = a.sparsity.shape[0]
            sparsity[os:os+m, :] = a.sparsity
        sparsity = sp.csr_matrix(sparsity)

        work_coeffs = len(args) # This is a varargs atom.
        return CoeffData(self, args, var,
                         macro_name = "hstack_coeffs",
                         sparsity = sparsity,
                         data = offsets,
                         work_coeffs = work_coeffs)
