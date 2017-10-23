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




class CodeGenerator:
    

    def __init__(self, objective, constraints,
                 solver=None,
                 include_solver=True):
        self.objective = objective.expr
        self.constraints = constraints
        self.template_vars = {}


    def codegen(self, target_dir,
                solver = None,
                include_solver = True,
                dump = False):

        if solver == None:
            solver = 'ecos'
        elif not (solver in SOLVER_INTFS.keys):
            raise TypeError("Unknown solver %s." % str(solver))

        # Create the target directory.
        make_target_dir(target_dir)

        # Create expression handler and solver interfaces.
        expr_handler = ExprHandler()
        solver_intf = SOLVER_INTFS[solver](expr_handler, include_solver)

        # Process objective, constraints.
        solver_intf.process_problem(self.objective, self.constraints)

        # Get template vars for the expr tree processor, then render.
        self.template_vars.update(expr_handler.get_template_vars())
        expr_handler.render(target_dir)

        # Get template variables from solver interface, then render.
        self.template_vars.update(solver_intf.get_template_vars())
        solver_intf.render(target_dir)

        # Get template variables to render template
        template_handler = TemplateHandler(target_dir)
        self.template_vars.update(template_handler.get_template_vars())
        template_handler.render(target_dir, self.template_vars)

        if dump:
            return self.template_vars
