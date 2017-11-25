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
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.hstack import Hstack, hstack

# QUADRATIC EXPR HANDLING
from cvxpy.utilities.replace_quad_forms import replace_quad_forms

from cvxpy_codegen.expr_handler.expr_handler import \
        ExprHandler, AffineOperator




EXPR_HANDLER_C_JINJA = 'expr_handler/tree/handler.c.jinja'


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



class TreeQuadForm():
    def __init__(self, name, quad_coeffs, x_length):
        self.name = name
        self.shape = (x_length, x_length)
        self.quad_coeffs = quad_coeffs
        self.sparsity = self._get_sparsity()
        self.work_float = max([qc.work_float for qc in quad_coeffs] + [0])

    def _get_sparsity(self):
        P = sp.lil_matrix(self.shape)
        for qc in self.quad_coeffs:
            offset = qc.var_offset
            if qc.is_elemwise:
                P_repeated = qc.P_data.storage.sparsity
                idxs = slice(offset, offset + P_repeated.shape[0])
            else:
                repeats = qc.shape[0]
                P_qf = qc.P_data.storage.sparsity
                P_repeated = repmat(P_qf, repeats, repeats)
                idxs = slice(offset, offset + P_qf.shape[0]*repeats)
            P[idxs,idxs] = P_repeated
        return sp.csc_matrix(sp.triu(P))

def repmat(P, m, n):
    l = []
    for i in range(m):
        l += [sp.hstack([P]*n)]
    return sp.vstack(l)



class QuadCoeff():
    def __init__(self, quad_form, coeff, P_data, var_offsets):
        self.coeff = coeff
        self.P_data = P_data
        self.shape = quad_form.shape
        if self.shape == ():
           self.shape = (1,1)

        # Get the variable offset for the argument variable of the quad_form.
        self.var_id = quad_form.args[0].id
        self.var_offset = var_offsets[self.var_id]

        # This workspace is required to build the coefficient in C.
        self.work_float = quad_form.size

        # CVXPY handles these two cases differently. Hopefully that changes.
        if coeff.sparsity.shape[1] > 1:
            self.is_elemwise = True
        else:
            self.is_elemwise = False




class TreeExprHandler(ExprHandler):

    def __init__(self, debug=False):

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
        self.template_vars = {'debug' : debug}

        # Operators.
        self.aff_operators = []
        self.aff_functionals = []
        self.quad_forms = []



    def aff_operator(self, exprs, name, x_length, var_offsets):
        expr_datas = []
        for e in exprs:
            expr_datas += [self._process_expression(e, var_offsets)]
        coeff = self._get_sparsity(expr_datas, x_length, var_offsets)
        op = TreeAffineOperator(name, coeff, exprs=expr_datas)
        self.aff_operators += [op]
        return op



    def aff_functional(self, expr, name, x_length, var_offsets):
        expr_data = self._process_expression(expr, var_offsets)
        coeff = self._get_sparsity([expr_data], x_length, var_offsets)
        op = TreeAffineOperator(name, coeff, exprs=[expr_data])
        self.aff_functionals += [op]
        return op



    # Return all variables needed to evaluate the C templates.
    def get_template_vars(self, x_length, var_offsets):

        # The sizes of the C workspaces needed to evaluate the
        # the different operations on linops coefficients.
        work_int     = max([c.work_int    for c in self.linop_coeffs] + 
                           [data.work_int for data in self.expressions] + [0])
        work_float   = max([c.work_float  for c in self.linop_coeffs] +
                           [data.work_float for data in self.expressions] +
                           [qf.work_float for qf in self.quad_forms] + [0])
        work_varargs = max([data.work_varargs for data in self.expressions] + [0])
        work_coeffs  = max([c.work_coeffs for c in self.linop_coeffs] + [0])

        # Get the named variables.
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
        
        unique_linops = {c.macro_name for c in self.linop_coeffs}
        unique_atoms = {c.macro_name for c in self.expressions}

        # Fill out the variables needed to render the C template.
        self.template_vars.update({'vars': self.vars,
                                   'named_vars': self.named_vars,
                                   'constants': self.constants,
                                   'named_params': self.named_params.values(),
                                   'callback_params': self.callback_params,
                                   'expressions': self.expressions,
                                   'linop_coeffs': self.linop_coeffs,
                                   'unique_linops': unique_linops,
                                   'unique_atoms': unique_atoms,
                                   'work_int': work_int,
                                   'work_float': work_float,
                                   'work_varargs': work_varargs,
                                   'work_coeffs': work_coeffs,
                                   'x_length': x_length,
                                   'zip': zip,
                                   'aff_operators': self.aff_operators,
                                   'aff_functionals': self.aff_functionals,
                                   'quad_forms': self.quad_forms,
                                   'var_offsets': var_offsets,
                                   'expr_handler_c_jinja': EXPR_HANDLER_C_JINJA,
                                   'has_matrix_vars'  : has_matrix_vars,
                                   'has_vector_vars'  : has_vector_vars,
                                   'CONST_ID' : CONST_ID })

        return self.template_vars



    # Recursively process an affine expression,
    # collecting data according to the operator type.
    def _process_expression(self, expr, var_offsets):
        expr_type = None
        expr_data = None
        is_linop = False

        if isinstance(expr, CallbackParam):
            data_arg = self._process_expression(expr.atom, var_offsets)
            data = CbParamData(expr, [data_arg])
            if expr.id not in self.cbparam_ids: # Check if already there.
                self.callback_params += [data]
                self.cbparam_ids += [expr.id]

        elif isinstance(expr, Parameter):
            if expr.id not in self.named_params: # Check if already there.
                data = ParamData(expr)
                self.named_params.update({expr.id: data})
            else:
                data = self.named_params[expr.id]

        elif isinstance(expr, Constant):
            data = ConstData(expr)
            self.constants += [data]

        elif isinstance(expr, Variable):
            if expr.id not in self.vars: # Check if already there.
                data = VarData(expr, var_offsets.get(expr.id))
                self.vars.update({expr.id : data})
            else:
                data = self.vars[expr.id]

        elif isinstance(expr, Atom):
            if id(expr) in self.linop_ids: # Check if already there.
                idx = self.linop_ids.index(id(expr))
                self.linops[idx].force_copy() # TODO makes unnecessary copies?
                data = self.linops[idx]
            else:
                expr = self.preprocess_expr(expr)
                arg_data = []
                for arg in expr.args: # Recurse on args.
                    arg_data += [self._process_expression(arg, var_offsets)]
                data = get_expr_data(expr, arg_data)
                #self.linop_coeffs += data.get_coeffs().values()
                #if data.has_offset:
                #    offset_expr = data.get_offset_expr()
                #    self.expressions += [offset_expr]
                #    if offset_expr.macro_name not in self.unique_atoms:
                #        self.unique_atoms += [offset_expr.macro_name]

                self.linops += [data]
                #for coeff in data.coeffs.values():
                #    if coeff.macro_name not in self.unique_linops:
                #        self.unique_linops += [coeff.macro_name]

        else:
            raise TypeError('Invalid expression tree type: %s' % type(expr))

        #self.linop_coeffs += data.get_coeffs().values()
        self.linop_coeffs += data.coeffs.values()
        print
        print type(data)
        print data.has_offset
        if data.has_offset:
            #offset_expr = data.get_offset_expr()
            offset_expr = data.offset
            self.expressions += [offset_expr]

        return data



    # TODO move in with atoms
    def preprocess_expr(self, expr):
        if isinstance(expr, AddExpression):
            new_args = []
            for a in expr.args:
                if np.prod(expr.shape) != 1 and np.prod(a.shape) == 1:
                    new_args += [promote(a, expr.shape)]
                else:
                    new_args += [a]
            expr.args = new_args
        if isinstance(expr, MulExpression) and not isinstance(expr, multiply):
            arg0 = expr.args[0]
            if (len(arg0.shape) == 1 and len(expr.args[1].shape) == 2):
                n = expr.args[0].shape[0]
                m = expr.args[1].shape[1]
                lhs = reshape(expr.args[0], (1, n))
                rhs = expr.args[1]
                expr = reshape(lhs * rhs, (m,))
            elif (len(arg0.shape) == 1 and len(expr.args[1].shape) == 1):
                n = expr.args[0].shape[0]
                lhs = reshape(expr.args[0], (1, n))
                rhs = reshape(expr.args[1], (n, 1))
                expr = reshape(lhs * rhs, ())
        if isinstance(expr, Hstack):
            if all([len(a.shape) == 2 for a in expr.args]):
                return expr
            elif all([len(a.shape) <= 1 for a in expr.args]):
                new_args = []
                for i, a in enumerate(expr.args):
                    new_args += [reshape(a, (1, a.size))]
                hstack_expr = hstack(new_args)
                expr = reshape(hstack_expr, (hstack_expr.size,))
            else:
                return Exception("Bad dimensions used in Hstack atom.")
        return expr



    def render(self, target_dir):
        render(target_dir, self.template_vars,
               EXPR_HANDLER_C_JINJA, 'expr_handler.c')



    # Gets the sparsity patterns of the coefficient.
    # (This tells us how much memory to allocate in C).
    def _get_sparsity(self, exprs, x_length, var_offsets):

        coeff = sp.csc_matrix((0, x_length), dtype=bool)
        for e in exprs:
            coeff = sp.vstack([coeff, e.get_matrix(x_length, var_offsets)])
        return sp.csc_matrix(coeff)



    def quad_functional(self, root, name, x_length, var_offsets):
       
        # Make sure root is not a quad_form:
        root += 0

        # Replace the quad_forms in root with dummy variables,
        # return a dict mapping dummy variable id to a tuple
        # of ( ).
        quad_forms = replace_quad_forms(root, {})
        dummy_var_ids = quad_forms.keys()

        # Process the root expression.
        expr_data = self._process_expression(root, var_offsets)

        # Get the coefficients of the root expression corresponding to the quad_forms.
        # (The remaining coefficients are the affine part of the quadratic
        # functional.)
        coeffs = expr_data.pop_coeffs(dummy_var_ids)

        # Process the affine part of the root expression.
        coeff = self._get_sparsity([expr_data], x_length, var_offsets)
        self.aff_functionals += [TreeAffineOperator('obj', coeff, exprs=[expr_data])]

        quad_coeffs = []
        for dummy_var_id, coeff in zip(dummy_var_ids, coeffs):

            # Get the quad_form.
            quad_form = quad_forms[dummy_var_id][2]

            # Process the quadratic coefficient of the quad_form.
            P_expr = quad_form.args[1]
            P_data = self._process_expression(P_expr, var_offsets)

            quad_coeffs += [QuadCoeff(quad_form, coeff, P_data, var_offsets)]

        quad_form = TreeQuadForm('quad', quad_coeffs, x_length)
        self.quad_forms += [quad_form]

        return quad_form.sparsity
