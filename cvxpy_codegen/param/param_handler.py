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

from cvxpy.lin_ops.lin_utils import get_expr_params
from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.atom import Atom
from cvxpy_codegen.atoms.atoms import get_atom_data
from cvxpy_codegen.param.expr_data import ParamData, ConstData, CbParamData
from cvxpy_codegen.utils.utils import render
import numpy

CBP_TO_SPARSITY = dict()





class ParamHandler():

    
    def __init__(self, objective, eq_constr, leq_constr):
        
        # Get the root parameters, ie, the roots of the expression trees
        # with only constant or parameters descendants.
        self.root_params = []
        self.root_params += get_expr_params(objective)
        for constr in eq_constr:
            self.root_params += get_expr_params(constr.expr)
        for constr in leq_constr:
            self.root_params += get_expr_params(constr.expr)

        self.named_params = []
        self.param_ids = []
        self.callback_params = []
        self.cbparam_ids = []
        self.constants = []
        self.expressions = []
        self.expr_ids = []
        self.unique_exprs = []

        self.template_vars = dict()


    def cbp2sparsity(self): # TODO make this not global
        #cbp2sparsity = dict()
        for cbp in self.callback_params:
            #cbp2sparsity[cbp.name] = cbp.sparsity
            CBP_TO_SPARSITY[cbp.cbp_name] = cbp.sparsity
        #return cbp2sparsity


    def get_template_vars(self):
        for root_param in self.root_params:
            self.process_expression(root_param)

        work_int = max([data.work_int for data in self.expressions] + [0])
        work_float = max([data.work_float for data in self.expressions] + [0])

        self.template_vars.update({'named_params': self.named_params,
                                   'constants': self.constants,
                                   'callback_params': self.callback_params,
                                   'expressions': self.expressions,
                                   'work_int': work_int,
                                   'work_float': work_float,
                                   'unique_exprs': self.unique_exprs})

        return self.template_vars



    def process_expression(self, expr):

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
                    data_list = get_atom_data(expr, arg_data)
                    self.expressions += data_list
                    data = data_list[-1]
                    self.expr_ids += [id(expr)]
                    for d in data_list:
                        if d.macro_name not in self.unique_exprs: # Check if already there.
                            self.unique_exprs += [d.macro_name]
        else:
            raise TypeError('Invalid expression tree type: %s' % type(expr))

        return data


    def render(self, target_dir):
        render(target_dir, self.template_vars, 'param/param.c.jinja', 'param.c')
