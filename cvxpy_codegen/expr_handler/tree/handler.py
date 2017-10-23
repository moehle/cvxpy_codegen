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

from cvxpy_codegen.object_data \
        import ParamData, ConstData, CbParamData, CONST_ID, CoeffData, VarData
from cvxpy_codegen.object_data.atom_data import AtomData
from cvxpy_codegen.object_data.constr_data import ConstrData
import scipy.sparse as sp
import numpy as np
from cvxpy_codegen.utils.utils import render
from cvxpy_codegen.atoms.atoms import get_expr_data

from cvxpy.lin_ops.lin_op import SCALAR_CONST, DENSE_CONST, SPARSE_CONST, PARAM, VARIABLE
from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.variable import Variable

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.promote import promote

from cvxpy_codegen.expr_handler.expr_handler import \
        ExprHandler, AffineOperator


class TreeAffineOperator(AffineOperator):
    def __init__(self, name, coeff, exprs):
        self.name = name
        self.coeff = coeff
        self.shape = coeff.shape
        self.exprs = exprs

        self.vert_offsets = []
        count = 0
        for e in exprs:
            self.vert_offsets += [count]
            count += e.length



class TreeExprHandler(ExprHandler):

    def __init__(self):

        # A list of all variables (not just the named ones).
        self.named_vars = []
        self.vars = []
        self.var_ids = []
        
        # List of Coefficients of linear expressions, corresponding to a variable.
        self.linop_coeffs = []

        # The parameters appearing in the linear operators.
        self.named_params = []
        self.param_ids = []

        # The callback parameters appearing in the linops.
        self.callback_params = []
        self.cbparam_ids = []

        # Constants appearing (as arguments) to the linops.
        self.constants = []

        # A list of the linear operators.
        self.linops = []
        self.linop_ids = []

        # The unique types of linops found in the linop trees.
        self.unique_linops = []

        # The atoms found in the linop trees.
        self.expressions = []
        self.expr_ids = []
        self.unique_atoms = []

        # The variables needed to render the C code.
        self.template_vars = dict()



    def aff_operator(self, exprs, name, x_length, var_offsets):
        expr_datas = []
        size = 0
        for e in exprs:
            expr_datas += [self.process_expression(e)]
            size += e.size

        coeff = self._get_sparsity(exprs, x_length, var_offsets)
        self.aff_operators += [TreeAffineOperator(name, coeff, exprs=expr_datas)]



    def aff_functional(self, expr, name, x_length, var_offsets):
        expr_datas = []
        size = 0
        for e in exprs:
            expr_datas += [self.process_expression(e)]
            size += e.size

        coeff = self._get_sparsity(exprs, x_length, var_offsets)
        self.aff_operators += [TreeAffineOperator(name, coeff, exprs=expr_datas)]



    # Return all variables needed to evaluate the C templates.
    def get_template_vars(self):

        # The sizes of the C workspaces needed to evaluate the
        # the different operations on linops coefficients.
        work_int     = max([c.work_int    for c in self.linop_coeffs] + 
                           [data.work_int for data in self.expressions] + [0])
        work_float   = max([c.work_float  for c in self.linop_coeffs] +
                           [data.work_float for data in self.expressions] + [0])
        work_varargs = max([data.work_varargs for data in self.expressions] + [0])
        work_coeffs  = max([c.work_coeffs for c in self.linop_coeffs] + [0])

        # Get the named variables.
        self.named_vars = [v for v in self.vars if v.is_named]
        if not self.named_vars:
            raise TypeError("Problem has no variables.")

        # Fill out the variables needed to render the C template.
        self.template_vars.update({'vars': self.vars,
                                   'named_vars': self.named_vars,
                                   'constants': self.constants,
                                   'named_params': self.named_params,
                                   'callback_params': self.callback_params,
                                   'expressions': self.expressions,
                                   'linop_coeffs': self.linop_coeffs,
                                   'unique_linops': self.unique_linops,
                                   'unique_atoms': self.unique_atoms,
                                   'work_int': work_int,
                                   'work_float': work_float,
                                   'work_varargs': work_varargs,
                                   'work_coeffs': work_coeffs,
                                   'CONST_ID' : CONST_ID })

        return self.template_vars



    # Recursively process a linear operator,
    # collecting data according to the operator type.
    def process_expression(self, expr):
        expr_type = None
        expr_data = None
        is_linop = False

        if isinstance(expr, CallbackParam):
            data_arg = self.process_expression(expr.atom)
            data = CbParamData(expr, [data_arg])
            if expr.id not in self.cbparam_ids: # Check if already there.
                self.callback_params += [data]
                self.cbparam_ids += [expr.id]

        elif isinstance(expr, Parameter):
            data = ParamData(expr)
            if expr.id not in self.param_ids: # Check if already there.
                self.named_params += [data]
                self.param_ids += [expr.id]

        elif isinstance(expr, Constant):
            data = ConstData(expr)
            self.constants += [data]

        elif isinstance(expr, Variable):
            data = VarData(expr)
            if expr.id not in self.var_ids: # Check if already there.
                self.vars += [data]
                self.var_ids += [expr.id]

        elif isinstance(expr, Atom):
            if id(expr) in self.linop_ids: # Check if already there.
                idx = self.linop_ids.index(id(expr))
                self.linops[idx].force_copy()
                data = self.linops[idx]
            else:
                self.preprocess_expr(expr)
                arg_data = []
                for arg in expr.args: # Recurse on args.
                    arg_data += [self.process_expression(arg)]
                data = get_expr_data(expr, arg_data)
                self.linop_coeffs += data.get_coeffs().values()
                if data.has_offset:
                    offset_expr = data.get_offset_expr()
                    self.expressions += [offset_expr]
                    if offset_expr.macro_name not in self.unique_atoms:
                        self.unique_atoms += [offset_expr.macro_name]

                self.linops += [data]
                for coeff in data.coeffs.values():
                    if coeff.macro_name not in self.unique_linops:
                        self.unique_linops += [coeff.macro_name]

        else:
            raise TypeError('Invalid expression tree type: %s' % type(expr))

        return data


    # Vstack the exprs into a matrix.
    def process_matrix(self, exprs):
        expr_datas = []
        for expr in exprs:
            expr_datas += [self.process_expr(expr)]
        self.expr_matrices += [ExprMatrix(expr_datas)]



    def preprocess_expr(self, expr):
        if isinstance(expr, AddExpression):
           new_args = []
           for a in expr.args:
               if np.prod(expr.shape) != 1 and np.prod(a.shape) == 1:
                   new_args += [promote(a, expr.shape)]
               else:
                   new_args += [a]
           expr.args = new_args




    def render(self, target_dir):
        render(target_dir, self.template_vars,
               'expr_handler/expr_handler.c.jinja', 'expr_handler.c')



    # Gets the sparsity patterns of the coefficient.
    # (This tells us how much memory to allocate in C).
    def _get_sparsity(self, exprs, x_length, var_offsets):

        coeff = sp.csc_matrix((0, x_length), dtype=bool)
        for e in exprs:
            coeff = sp.vstack([coeff, e.get_matrix(x_length, var_offsets)])
        return sp.csc_matrix(coeff)
