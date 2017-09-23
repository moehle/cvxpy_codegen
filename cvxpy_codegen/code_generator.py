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
from cvxpy_codegen.solvers.solver_intfs import SOLVER_INTFS
from cvxpy_codegen.utils.utils import make_target_dir
import cvxpy.settings as s
import numpy as np

from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.matrix_stuffing import MatrixStuffing
from cvxpy.reductions import InverseData


def codegen(prob, target_dir, **kwargs):
    cg = CodeGenerator(prob, **kwargs)
    return cg.codegen(target_dir)




class CodeGenerator:
    

    def __init__(self, prob,
                 solver=None,
                 include_solver=True,
                 inv_data = None,
                 dump = False):

        # TODO: make nicer
        sc = construct_solving_chain(prob, solver="ECOS")
        for r in sc.reductions:
            if isinstance(r, Dcp2Cone):
                prob, dcp2cone_inv_data = r.apply(prob)
            if isinstance(r, CvxAttr2Constr):
                prob, attr_inv_data = r.apply(prob)

        self.objective = prob.objective.expr
        self.constraints = prob.constraints
        if not inv_data:
            inv_data = InverseData(prob)
        self.inv_data = inv_data
        self.template_vars = {}

        self.include_solver = include_solver
        self.dump = dump
        self.solver = solver

        



    def codegen(self, target_dir):
        
        solver = self.solver
        dump = self.dump
        include_solver = self.include_solver

        if solver == None:
            solver = 'ecos'
        elif not (solver in SOLVER_INTFS.keys()):
            raise TypeError("Unknown solver %s." % str(solver))

        # Create the target directory.
        make_target_dir(target_dir)

        # Create expression handler and solver interfaces.
        expr_handler = ExprHandler()
        solver_intf = SOLVER_INTFS[solver](expr_handler, self.inv_data, include_solver)

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
