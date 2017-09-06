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

from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy_codegen.object_data import ParamData, ConstData, CbParamData, CONST_ID, LinOpCoeffData, VarData
from cvxpy_codegen.object_data.linop_data import LinOpData
from cvxpy.lin_ops.lin_op import SCALAR_CONST, DENSE_CONST, SPARSE_CONST, PARAM, VARIABLE
from cvxpy_codegen.linop.constr_data import ConstrData
import scipy.sparse as sp
from cvxpy_codegen.utils.utils import render

from cvxpy.lin_ops.lin_op import LinOp
from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.atom import Atom



class ExprHandler():

    
    def __init__(self, sym_data, objective, eq_constr, leq_constr):

        # A list of all variables (not just the named ones).
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

        # Symbolic problem data.
        self.sym_data = sym_data

        # The variables needed to render the C code.
        self.template_vars = dict()

        # Process all the linear expression trees to fill out the
        # lists given above.
        self.objective  = self.process_expression(objective)
        self.eq_constr  = self.process_constr(eq_constr)
        self.leq_constr = self.process_constr(leq_constr)



    # For each constraint, collect data on the constraint,
    # then process its expression trees.
    def process_constr(self, constrs):
        constr_data = []
        vert_offset = 0
        for constr in constrs:
            expr_data = self.process_expression(constr.expr)
            constr_data += [ConstrData(constr, expr_data, vert_offset=vert_offset)]
            vert_offset += expr_data.size[0] * expr_data.size[1] 
        return constr_data


    # Return all variables needed to evaluate the C templates.
    def get_template_vars(self):

        # The sizes of the C workspaces needed to evaluate the
        # the different operations on linops coefficients.
        work_int     = max([c.work_int    for c in self.linop_coeffs] + 
                           [data.work_int for data in self.expressions] + [0])
        work_float   = max([c.work_float  for c in self.linop_coeffs] +
                           [data.work_float for data in self.expressions] + [0])
        work_varargs = max([data.work_varargs for data in self.expressions] + [0])
        work_coeffs  = max([c.work_coeffs for c in self.linop_coeffs])

        # Recover the sparsity patterns of the coefficient matrices.
        obj_coeff, eq_coeff, leq_coeff = self.get_sparsity()

        # Fill out the variables needed to render the C template.
        self.template_vars.update({'vars': self.vars,
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
                                   'objective': self.objective,
                                   'eq_constr': self.eq_constr,
                                   'leq_constr': self.leq_constr,
                                   'obj_coeff': obj_coeff,
                                   'eq_coeff': eq_coeff,
                                   'leq_coeff': leq_coeff,
                                   'var_offsets' : self.sym_data.var_offsets,
                                   'CONST_ID' : CONST_ID,
                                   'x_length' : self.sym_data.x_length })

        return self.template_vars



    # Recursively process a linear operator,
    # collecting data according to the operator type.
    def process_expression(self, expr):
        if isinstance(expr, LinOp):
            expr_type = expr.type 
            expr_data = expr.data
            is_linop = True
        else:
            expr_type = None
            expr_data = None
            is_linop = False


        if isinstance(expr, CallbackParam) or \
                  (expr_type == PARAM and isinstance(expr_data, CallbackParam)):
            data_arg = self.process_expression(expr.atom)
            data = CbParamData(expr, [data_arg])
            if expr.id not in self.cbparam_ids: # Check if already there.
                self.callback_params += [data]
                self.cbparam_ids += [expr.id]

        elif isinstance(expr, Parameter) or \
                  (expr_type == PARAM and isinstance(expr_data, Parameter)):
            data = ParamData(expr)
            if expr.id not in self.param_ids: # Check if already there.
                self.named_params += [data]
                self.param_ids += [expr.id]

        elif isinstance(expr, Constant) or \
                  expr_type in [SCALAR_CONST, DENSE_CONST, SPARSE_CONST]:
            data = ConstData(expr)
            self.constants += [data]

        elif expr_type == VARIABLE:
            data = VarData(expr)
            if expr.data not in self.var_ids: # Check if already there.
                self.vars += [data]
                self.var_ids += [expr.data]

        elif isinstance(expr, Atom):
            if not expr.parameters(): # expr is just a constant without parameters.
                data = ConstData(Constant(expr.value))
                self.constants += [data]
            else: # Recurse on arguments:
                if id(expr) in self.expr_ids: # Check if already there.
                    idx = self.expr_ids.index(id(expr))
                    self.expressions[idx].force_copy()
                    data = self.expressions[idx]
                else:
                    arg_data = []
                    for arg in expr.args:
                        arg_data += [self.process_expression(arg)]
                    data = get_atom_data(expr, arg_data)
                    self.expressions += [data]
                    self.expr_ids += [id(expr)]
                    if data.macro_name not in self.unique_exprs:
                            self.unique_exprs += [data.macro_name]

        elif is_linop is True:  # expr is a linear operator expression:
            if id(expr) in self.linop_ids: # Check if already there.
                idx = self.linop_ids.index(id(expr))
                self.expressions[idx].force_copy()
                data = self.expressions[idx]
            else:
                arg_data = []
                for arg in expr.args: # Recurse on args.
                    arg_data += [self.process_expression(arg)]
                data = LinOpData(expr, arg_data)
                self.linop_coeffs += data.coeffs.values() # Coefficients of variables.
                if data.has_offset:
                    self.expressions += [data.offset_expr] # Expressions of constants.
                self.linops += [data]
                for coeff in data.coeffs.values():
                    if coeff.macro_name not in self.unique_linops:
                         self.unique_linops += [coeff.macro_name]

        else:
            raise TypeError('Invalid expression tree type: %s' % type(expr))

        return data



    # Gets the sparsity patterns of the objective and constraint coefficients.
    # (This tells us how much memory to allocate in C).
    def get_sparsity(self):

        # Get Boolean sparse matrix for the objective.
        obj_coeff = self.objective.get_matrix(self.sym_data)
        
        # Get Boolean sparse matrix for the equality constraints.
        eq_coeff = sp.csc_matrix((0,self.sym_data.x_length), dtype=bool)
        for c in self.eq_constr:
            eq_coeff = sp.vstack([eq_coeff, c.get_matrix(self.sym_data)])

        # Get Boolean sparse matrix for the inequality constraints.
        leq_coeff = sp.csc_matrix((0,self.sym_data.x_length), dtype=bool)
        for c in self.leq_constr:
            leq_coeff = sp.vstack([leq_coeff, c.get_matrix(self.sym_data)])

        return (sp.csc_matrix(obj_coeff),
                sp.csc_matrix(eq_coeff),
                sp.csc_matrix(leq_coeff))



    def render(self, target_dir):
        render(target_dir, self.template_vars,
               'expr_handler/expr_handler.c.jinja', 'expr_handler.c')
