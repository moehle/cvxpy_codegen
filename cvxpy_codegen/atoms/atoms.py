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
from cvxpy_codegen.object_data.aff_atom_data import AffAtomData


from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.atoms.affine.diag import diag_vec, diag_mat
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.multiply import multiply
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.vstack import Vstack
from cvxpy.atoms import *


from cvxpy_codegen.atoms.abs.abs import AbsData
from cvxpy_codegen.atoms.add.add import AddData
from cvxpy_codegen.atoms.diag_vec.diag_vec import DiagVecData
from cvxpy_codegen.atoms.diag_mat.diag_mat import DiagMatData
from cvxpy_codegen.atoms.hstack.hstack import HStackData
from cvxpy_codegen.atoms.index.index import IndexData
from cvxpy_codegen.atoms.kron.kron import KronData
from cvxpy_codegen.atoms.multiply.multiply import MultiplyData
from cvxpy_codegen.atoms.mul.mul import MulData
from cvxpy_codegen.atoms.max.max import MaxData
from cvxpy_codegen.atoms.neg.neg import NegData
from cvxpy_codegen.atoms.pos.pos import PosData
from cvxpy_codegen.atoms.promote.promote import PromoteData
from cvxpy_codegen.atoms.reshape.reshape import ReshapeData
from cvxpy_codegen.atoms.sum.sum import SumData
from cvxpy_codegen.atoms.trace.trace import TraceData
from cvxpy_codegen.atoms.transpose.transpose import TransposeData
from cvxpy_codegen.atoms.vstack.vstack import VStackData



GET_EXPR_DATA = {abs           : AbsData,
                 AddExpression : AddData,
                 diag_mat      : DiagMatData,
                 diag_vec      : DiagVecData,
                 Hstack        : HStackData,
                 index         : IndexData,
                 kron          : KronData,
                 max           : MaxData,
                 MulExpression : MulData,
                 multiply      : MultiplyData,
                 NegExpression : NegData,
                 pos           : PosData,
                 Promote       : PromoteData,
                 reshape       : ReshapeData,
                 Sum           : SumData,
                 trace         : TraceData,
                 transpose     : TransposeData,
                 Vstack        : VStackData }


def get_expr_data(expr, arg_data):
    if not type(expr) in GET_EXPR_DATA.keys():
        raise TypeError("Expressions involving "
                        "the %s atom not supported." % str(type(expr)))
    return GET_EXPR_DATA[type(expr)](expr, arg_data)
