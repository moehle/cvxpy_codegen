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

from cvxpy_codegen.object_data import ParamData, ConstData, CbParamData, CONST_ID, CoeffData, VarData
from cvxpy_codegen.object_data.atom_data import AtomData
from cvxpy_codegen.object_data.constr_data import ConstrData
import scipy.sparse as sp
from cvxpy_codegen.utils.utils import render
from cvxpy_codegen.atoms.atoms import get_expr_data

from cvxpy.lin_ops.lin_op import SCALAR_CONST, DENSE_CONST, SPARSE_CONST, PARAM, VARIABLE
from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.variable import Variable



class ExprHandler():

    
    def __init__(self):




    # Return all variables needed to evaluate the C templates.
    def get_template_vars(self):


    def render(self, target_dir):
        render(target_dir, self.template_vars,
               'expr_handler/expr_handler.c.jinja', 'expr_handler.c')
