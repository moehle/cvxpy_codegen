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

from cvxpy_codegen.expr_handler import ExprHandler
from cvxpy_codegen.templates.template_handler import TemplateHandler
#from cvxpy.problems.problem_data.sym_data import SymData
from cvxpy_codegen.solvers.solver_intfs import SOLVER_INTFS
from cvxpy_codegen.utils.utils import make_target_dir
import cvxpy.settings as s
import numpy as np





# TODO this doesn't seem to work, and should maybe be moved elsewhere
def cprint_var(var):
    s = ""
    for i in range(var.size[0]):
        for j in range(var.size[1]):
            s += '"  printf(\%s[\%d][\%d] = \%f\n", vars.%s[%d][%d];' % (var.name(), i, j)
    return s



class CodeGenerator:
    

    def __init__(self, objective, constraints, vars, params,
                 inv_data, solver=None):
        if vars == []:
            raise TypeError("Problem has no variables.")

        self.objective = objective.expr
        self.constraints = constraints
        self.inv_data = inv_data
        self.named_vars = self.get_named_vars(vars) # TODO need?
        self.params = self.get_named_params(params)

        offset = 0
        self.var_offsets = {}
        for var in vars:
            self.var_offsets.update({var.id : offset})
            offset += var.size
        self.x_length = offset

        # New expression handler.
        self.expr_handler = ExprHandler()

        if solver == None:
            solver = 'ecos'
        elif not (solver in SOLVER_INTFS.keys):
            raise TypeError("Unknown solver %s." % str(solver))
        self.solver = SOLVER_INTFS[solver](self.expr_handler, self.x_length, self.var_offsets)

        # TODO rm params, so all params hanlded by the param_handler:
        self.template_vars = {'named_vars' : self.named_vars,
                              'params' : self.params,
                              'cprint_var' : cprint_var,
                              'var_offsets' : self.var_offsets,
                              #'id2var' : self.id2var,
                              'x_length' : self.x_length}


    @staticmethod
    def get_named_vars(vars):
        named_vars = []
        names = []
        for var in vars:
            if var.name() in names:
                raise TypeError('Duplicate variable name %s' % var.name())
            #named_vars += [var]
            #names += var.name()
            if not var.name() == "%s%d" % (s.VAR_PREFIX, var.id): # TODO option to not ignore default names.
                named_vars += [var]
        return named_vars


    @staticmethod
    def get_named_params(params):
        named_params = []
        names = []
        for param in params:
            if param.name() in names:
                raise TypeError('Duplicate parameter name %s' % param.name())
            named_params += [param]
            names += param.name()
        return named_params
        

    def codegen(self, target_dir):

        make_target_dir(target_dir)

        # Add solver to template variables.
        self.template_vars.update({'solver' : self.solver}) # TODO add back in
        self.template_vars.update({'solver_name' : self.solver.name}) # TODO add back in

        # Get template variables from solver, then render.
        self.solver.process_problem(self.objective, self.constraints)
        self.template_vars.update(self.solver.get_template_vars(
                self.inv_data, self.template_vars))
        self.solver.render(target_dir)

        # Get template vars for the expr tree processor, then render.
        self.template_vars.update(self.expr_handler.get_template_vars())
        self.expr_handler.render(target_dir)

        # Get template variables to render template
        template_handler = TemplateHandler(target_dir)
        self.template_vars.update(template_handler.get_template_vars())
        template_handler.render(target_dir, self.template_vars)
