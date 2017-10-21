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

from cvxpy.lin_ops.lin_op import SCALAR_CONST, DENSE_CONST, SPARSE_CONST, PARAM, VARIABLE
from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.variable import Variable

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.promote import promote
from cvxpy_codegen.expr_handler.explicit.sym_expr \
        import SymAdd, SymMult, SymDiv, SymParam, SymConst

import cvxpy_codegen.expr_handler.explicit.sym_matrix as sym 
from cvxpy_codegen.expr_handler.explicit.atom_to_coeffs import \
        get_coeffs, const_mat, mul_by_const

from cvxpy_codegen.expr_handler.expr_handler import \
        ExprHandler, AffineOperator

from cvxpy_codegen.utils.utils import CONST_ID




class ExplicitExprHandler(ExprHandler):
    
    def __init__(self, var_offsets):

        # A list of all variables (not just the named ones).
        self.named_vars = []
        self.vars = {}
        
        # List of Coefficients of linear expressions, corresponding to a variable.
        self.linop_coeffs = []

        # The parameters appearing in the linear operators.
        self.named_params = {}

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

        # Affine operators:
        self.aff_operators = []
        self.aff_functionals = []

        # Store variable offsets:
        self.var_offsets = var_offsets



    def aff_operator(self, exprs, name, x_length, var_offsets):
        A, b = self._exprs_to_mat(exprs, x_length, var_offsets)
        b = b.as_vector()
        self.aff_operators += [AffineOperator(A, b, name)]


    def aff_functional(self, expr, name, x_length, var_offsets):
        c, d = self._exprs_to_mat([expr], x_length, var_offsets)
        c = sym.transpose(c).as_vector()
        d = d.as_vector()
        self.aff_functionals += [AffineOperator(c, d, name)]



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

        self.named_vars = [v for v in self.vars.values() if v.is_named]

        var_names = []
        for v in self.named_vars:
            if v.name in var_names:
                raise Exception('Duplicate variable name "%s".' % v.name)
            var_names += v.name

        param_names = []
        for p in self.named_params.values():
            if p.name in param_names:
                raise Exception('Duplicate parameter name "%s".' % p.name)
            param_names += p.name

        has_vector_vars = any([v.is_vector() for v in self.named_vars])
        has_matrix_vars = any([v.is_matrix() for v in self.named_vars])

        # Fill out the variables needed to render the C template.
        self.template_vars.update({'vars': self.vars,
                                   'named_vars': self.named_vars,
                                   'constants': self.constants,
                                   'named_params': self.named_params.values(),
                                   'callback_params': self.callback_params,
                                   'expressions': self.expressions,
                                   'linop_coeffs': self.linop_coeffs,
                                   'unique_linops': self.unique_linops,
                                   'unique_atoms': self.unique_atoms,
                                   'work_int': work_int,
                                   'work_float': work_float,
                                   'work_varargs': work_varargs,
                                   'work_coeffs': work_coeffs,
                                   'aff_operators': self.aff_operators,
                                   'aff_functionals': self.aff_functionals,
                                   'SymAdd'    : SymAdd,
                                   'SymMult'   : SymMult,
                                   'SymDiv'    : SymDiv,
                                   'SymParam'  : SymParam,
                                   'SymConst'  : SymConst,
                                   'Variable'  : Variable,
                                   'has_matrix_vars'  : has_matrix_vars,
                                   'has_vector_vars'  : has_vector_vars,
                                   'CONST_ID' : CONST_ID })

        return self.template_vars



    def render(self, target_dir):
        render(target_dir, self.template_vars,
               'expr_handler/explicit/handler.c.jinja', 'handler.c')



    # Recursively process a linear operator,
    # collecting data according to the operator type.
    def _process_expression(self, expr, update_vars=True):

        if isinstance(expr, CallbackParam):
            coeffs = self._process_expression(expr.atom)

        elif isinstance(expr, Parameter):
            if expr.id not in self.named_params: # Check if already there.
                data = ParamData(expr)
                self.named_params.update({expr.id: data})
            else:
                data = self.named_params[expr.id]
            coeffs = [(CONST_ID, sym.as_sym_matrix(expr).as_vector())]

        elif isinstance(expr, Constant):
            mat = sp.lil_matrix(expr.value)
            size = np.prod(mat.shape)
            mat = sp.csc_matrix(mat.T.reshape((1, size), order='C').T)
            coeffs = [(CONST_ID, mat)]

        elif isinstance(expr, Variable):
            if update_vars is True:
                if expr.id not in self.vars: # Check if already there.
                    data = VarData(expr, self.var_offsets[expr.id])
                    self.vars.update({expr.id : data})
                else:
                    data = self.vars[expr.id]
            coeffs = [(expr.id, sp.eye(np.prod(expr.shape)).tocsc())]

        elif isinstance(expr, Atom):
            self._preprocess_expr(expr)
            arg_coeff_list = []
            for arg in expr.args:
                arg_coeff_list += [self._process_expression(arg)]
            coeffs = get_coeffs(expr, arg_coeff_list)

        else:
            raise TypeError('Invalid expression tree type: %s' % type(expr))
        coeffs = [(c[0], sym.as_sym_matrix(c[1])) for c in coeffs]

        return coeffs


    def _preprocess_expr(self, expr):
        if isinstance(expr, AddExpression):
           new_args = []
           for a in expr.args:
               if np.prod(expr.shape) != 1 and np.prod(a.shape) == 1:
                   new_args += [promote(a, expr.shape)]
               else:
                   new_args += [a]
           expr.args = new_args



    def _exprs_to_mat(self, exprs, x_length, var_offsets):
        constr_size = int(sum([np.prod(e.shape) for e in exprs]))
        matrix = sym.zeros(constr_size, x_length)
        offset = sym.zeros(constr_size, 1)
        vert_offset = 0

        for e in exprs:
            c_mat, c_off = self._process_expr(e, x_length, var_offsets,
                               vert_offset=vert_offset, vert_size=constr_size)
            matrix += c_mat
            offset += c_off
            vert_offset += int(np.prod(e.shape))
        return matrix, offset
        


    # Vstack the exprs into a matrix.
    def _process_expr(self, expr, x_length, var_offsets, vert_offset=0, vert_size=1):

        #coeffs = get_coeffs(expr)
        coeffs = self._process_expression(expr)

        # TODO remove vert_size, replace with np.prod(expr.shape)
        offset = sym.zeros(vert_size, 1)
        matrix = sym.zeros(vert_size, x_length)
        for id_, sym_mat in coeffs:
            vert_start = vert_offset 
            vert_end = vert_start + np.prod(expr.shape)
            if id_ is CONST_ID: 
                sym_mat = sym.zero_pad(sym_mat, (vert_size, 1),
                                       (vert_start, 0))
                offset += sym_mat
            else: 
                horiz_offset = var_offsets[id_]
                sym_mat = sym.zero_pad(sym_mat, (vert_size, x_length),
                                       (vert_start, horiz_offset))
                matrix += sym_mat

        return matrix, offset



    # TODO finish
    #def process_quad_expr(self, root):
    #    quad_forms = replace_quad_forms(root, {})

    #    coeffs = self.process_expression(root, update_vars=False)
    #    
    #    for var_id in quad_forms:

    #    return



