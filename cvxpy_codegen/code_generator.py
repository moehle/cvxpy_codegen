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

from cvxpy_codegen.expr_handler.explicit.handler import ExplicitExprHandler
from cvxpy_codegen.expr_handler.tree.handler import TreeExprHandler
from cvxpy_codegen.templates.template_handler import TemplateHandler
from cvxpy_codegen.solvers.solver_intfs import SOLVER_INTFS
from cvxpy_codegen.utils.utils import make_target_dir
import cvxpy.settings as s
import numpy as np

from cvxpy.reductions import InverseData


def codegen(prob, target_dir,
            solver = None,
            include_solver = True,
            inv_data = None,
            codegen_mode = 'explicit',
            dump = False):

        # Solver defaults to ECOS.
        if solver == None:
            solver = "ecos"
        if not solver.lower() in SOLVER_INTFS:
            raise Exception('Solver "%s" not found' % solver)

        if not inv_data:
            inv_data = InverseData(prob)
        
        # Create the target directory.
        make_target_dir(target_dir)

        # Create expression handler and solver interfaces.
        if codegen_mode == 'explicit':
            expr_handler = ExplicitExprHandler(inv_data.var_offsets)
        elif codegen_mode == 'tree':
            expr_handler = TreeExprHandler(inv_data.var_offsets)
        else:
            raise Exception('Code generation mode "%s" not recognized.' % codegen_mode)

        # Process problem.
        solver_intf = SOLVER_INTFS[solver.lower()](expr_handler,
                            inv_data, include_solver)
        prob = solver_intf.preprocess_problem(prob)
        solver_intf.process_problem(prob)

        # Get template variables from solver interface, then render.
        template_vars = {}
        template_vars.update(solver_intf.get_template_vars())
        solver_intf.render(target_dir)

        # Get template vars for the expr tree processor, then render.
        template_vars.update(expr_handler.get_template_vars())
        expr_handler.render(target_dir)

        # Get template variables to render template
        template_handler = TemplateHandler(target_dir)
        template_vars.update(template_handler.get_template_vars())
        template_handler.render(target_dir, template_vars)

        if dump:
            return template_vars
