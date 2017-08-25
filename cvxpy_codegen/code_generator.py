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

from cvxpy_codegen.param.param_handler import ParamHandler
from cvxpy_codegen.templates.template_handler import TemplateHandler
from cvxpy.problems.problem_data.sym_data import SymData
from cvxpy_codegen.linop_sym.linop_handler_sym import LinOpHandlerSym
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
    

    def __init__(self, objective, constraints, vars, params, solver=None):
        if vars == []:
            raise TypeError("Problem has no variables.")

        if solver == None:
            solver = 'ecos'
        elif not (solver in SOLVER_INTFS.keys):
            raise TypeError("Unknown solver %s." % str(solver))
        self.solver = SOLVER_INTFS[solver]()

        self.objective = objective
        self.constraints = constraints
        self.sym_data = SymData(self.objective, self.constraints, self.solver.CVXPY_SOLVER)
        self.named_vars = self.get_named_vars(vars)
        self.params = self.get_named_params(params)

        # TODO rm params, so all params hanlded by the param_handler:
        self.template_vars = {'named_vars' : self.named_vars,
                              'params' : self.params,
                              'cprint_var' : cprint_var}



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
        self.template_vars.update({'solver' : self.solver})
        self.template_vars.update({'solver_name' : self.solver.name})
        self.template_vars.update({'sym_data' : self.sym_data})

        # get objective and linear constraint lists
        objective = self.sym_data.objective
        eq_constr, leq_constr, __ \
                 = self.solver.CVXPY_SOLVER.split_constr(self.sym_data.constr_map)

        # Get template variables from parameter handler, then render C file.
        param_handler = ParamHandler(objective, eq_constr, leq_constr)
        self.template_vars.update(param_handler.get_template_vars())
        param_handler.render(target_dir)
        param_handler.cbp2sparsity() # TODO remove somehow?

        # Get template vars for the linear expr tree processor, then render.
        linop_handler = LinOpHandlerSym(self.sym_data, objective, eq_constr, leq_constr)
        self.template_vars.update(linop_handler.get_template_vars())
        linop_handler.render(target_dir)

        # Get template variables from solver, then render.
        self.template_vars.update(self.solver.get_template_vars(
                self.sym_data, self.template_vars))
        self.solver.render(target_dir)

        # Get template variables to render template
        template_handler = TemplateHandler(target_dir)
        self.template_vars.update(template_handler.get_template_vars())
        template_handler.render(target_dir, self.template_vars)
