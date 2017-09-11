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

from cvxpy_codegen.code_generator import CodeGenerator
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr


def codegen(prob, target_dir, dump=False):

    vars = prob.variables()
    params = prob.parameters()

    # TODO: make nicer
    sc = construct_solving_chain(prob, solver="ECOS")
    for r in sc.reductions:
        if isinstance(r, Dcp2Cone):
            prob, dcp2cone_inv_data = r.apply(prob)
        if isinstance(r, CvxAttr2Constr):
            prob, __ = r.apply(prob)
    obj = prob.objective
    constraints = prob.constraints

    cg = CodeGenerator(obj, constraints, vars,
                       params, dcp2cone_inv_data)
    cg.codegen(target_dir)

    if dump:
        return cg.template_vars
