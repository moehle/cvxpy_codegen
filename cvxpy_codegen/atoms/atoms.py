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

from cvxpy_codegen.atoms.abs import *
from cvxpy_codegen.atoms.add import *
from cvxpy_codegen.atoms.diag_vec import *
from cvxpy_codegen.atoms.diag_mat import *
from cvxpy_codegen.atoms.hstack import *
from cvxpy_codegen.atoms.index import *
from cvxpy_codegen.atoms.max_entries import *
from cvxpy_codegen.atoms.mul import *
from cvxpy_codegen.atoms.mul_elemwise import *
from cvxpy_codegen.atoms.neg import *
from cvxpy_codegen.atoms.pos import *
from cvxpy_codegen.atoms.reshape import *
from cvxpy_codegen.atoms.trace import *
from cvxpy_codegen.atoms.vstack import *


from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.diag import diag_vec, diag_mat
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.mul_elemwise import mul_elemwise
from cvxpy.atoms import *



GET_ATOMDATA = {MulExpression : getdata_mul,
                AddExpression : getdata_add,
                pos           : getdata_pos,
                vstack        : getdata_vstack,
                hstack        : getdata_hstack,
                mul_elemwise  : getdata_mul_elemwise,
                index         : getdata_index,
                diag_vec      : getdata_diag_vec,
                diag_mat      : getdata_diag_mat,
                NegExpression : getdata_neg,
                reshape       : getdata_reshape,
                abs           : getdata_abs,
                trace         : getdata_trace,
                max_entries   : getdata_max_entries }


def get_atom_data(expr, arg_data):
    if not type(expr) in GET_ATOMDATA.keys():
        raise TypeError("Constant expressions involving the %s atom not supported." % str(type(expr)))
    return GET_ATOMDATA[type(expr)](expr, arg_data)
