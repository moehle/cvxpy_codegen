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

from jinja2 import Environment, PackageLoader, contextfilter
from cvxpy_codegen.utils.utils import render, PKG_PATH, EXP_CONE_LENGTH, call_macro
from cvxpy_codegen.solvers.embedded_solver_intf import EmbeddedSolverIntf
import os
import shutil as sh
import cvxpy.settings as s
from glob import glob
import ecos
from cvxpy.reductions.solvers.solver import group_constraints
from cvxpy.constraints import SOC, ExpCone, NonPos, Zero
from cvxpy_codegen.expr_handler.expr_handler import ExprHandler
from cvxpy_codegen.object_data.constr_data import NonPosData, SocData, ExpConeData, ZeroData
import scipy.sparse as sp
import numpy as np
from cvxpy import vstack, reshape

import osqp

from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp



class OsqpIntf(EmbeddedSolverIntf):
    name = 'osqp'
    SOLVER_TEMPLATE = "solvers/osqp_intf.c.jinja"

    # Macros to be used for the Python interface.
    DEFINE_MACROS = [('PYTHON',None),
                     ('DLONG', None),
                     ('LDL_LONG', None),
                     ('CTRLC', 1)]

    INCLUDE_DIRS = ['osqp/include']
    INCLUDES = "".join(["INCLUDES ="] + [" -I"+d for d in INCLUDE_DIRS])

    LIB_DIR = 'osqp/build/out/libemosqpstatic.a'
    LD_FLAGS = '-lemosqpstatic -Losqp/build/out'

    BUILD = 'osqp/build/out/libemosqpstatic.a:\n\tcd osqp/build && make'
    CLEAN = 'cd osqp/build && make clean'

    SOURCES = (['osqp/src/osqp/auxil.c',
                'osqp/src/osqp/ldl.c',
                'osqp/src/osqp/osqp.c',
                'osqp/src/osqp/proj.c',
                'osqp/src/osqp/util.c',
                'osqp/src/osqp/kkt.c',
                'osqp/src/osqp/lin_alg.c',
                'osqp/src/osqp/private.c',
                'osqp/src/osqp/scaling.c'])

    template_vars = { 'solver_sources'          : SOURCES,
                      'solver_include_dirs'     : INCLUDE_DIRS,
                      'solver_define_macros'    : DEFINE_MACROS,
                      'solver_template'         : SOLVER_TEMPLATE,
                      'LD_FLAGS'                : LD_FLAGS,
                      'LIB_DIR'                 : LIB_DIR,
                      'CLEAN'                   : CLEAN,
                      'BUILD'                   : BUILD,
                      'INCLUDES'                : INCLUDES,
                      'solver_name'             : name } 

    
    def __init__(self, include_solver=True):
        self.eq_constrs = []
        self.leq_constrs = []
        self.n_eq = 0
        self.n_leq = 0
        self.n_soc = 0
        self.n_exp = 0
        self.soc_sizes = []
        self.cone_dim = 0
        self.template_vars.update({'include_solver' : include_solver})
        self.P_sparsity = None
        self.A_sparsity = None
        self.include_solver = include_solver


    # TODO untested:
    def preprocess_problem(self, prob):
        sc = construct_solving_chain(prob, solver="OSQP")

        for r in sc.reductions:
            if isinstance(r, CvxAttr2Constr):
                prob, attr_inv_data = r.apply(prob)
                if not attr_inv_data == ():
                    id2new_var, id2old_var, __ = attr_inv_data
                    for id in id2old_var.keys():
                        # TODO Sets a private attribute :(
                        id2new_var[id]._name = id2old_var[id].name()
            if isinstance(r, Qp2SymbolicQp):
                prob, qp2sym_inv_data = r.apply(prob)

        return prob


    def get_template_vars(self):

        # Recover the sparsity patterns of the coefficient matrices.
        x_length = self.inv_data.x_length
        var_offsets = self.inv_data.var_offsets

        # TODO rename many of these:
        self.template_vars.update({
                'eq_dim'       : self.n_eq,
                'leq_dim'      : self.n_leq,
                'var_offsets'  : var_offsets,
                'x_length'     : x_length})

        return self.template_vars


    def render(self, target_dir):
        solver_dir = os.path.join(target_dir, 'osqp')
        
        # make target directory
        if os.path.exists(solver_dir):
            sh.rmtree(solver_dir)

        if self.include_solver:
            # Generate OSQP code.
            osqp_model = osqp.OSQP()

            m, n = self.A_sparsity.shape
            A = self.A_sparsity
            P = self.P_sparsity
            q = np.zeros(n)
            l = np.zeros(m)
            u = np.zeros(m)

            osqp_model.setup(P, q, A, l, u)
            osqp_model.codegen(solver_dir,
                               force_rewrite=True,
                               project_type='Makefile',
                               parameters='matrices')

        
        # Render interface template:
        render(target_dir, self.template_vars, 'solvers/osqp_intf.c.jinja', 'solver_intf.c')


    def process_problem(self, prob, expr_handler, inv_data):
        obj = prob.objective.expr
        constrs = prob.constraints

        self.expr_handler = expr_handler
        self.inv_data = inv_data

        # Separate constraints.
        constr_map = group_constraints(constrs)
        x_length = self.inv_data.x_length
        var_offsets = self.inv_data.var_offsets

        self.P_sparsity = self.expr_handler.quad_functional(
                obj, 'obj', x_length, var_offsets)

        exprs = []
        eq_constrs = constr_map[Zero]
        for constr in eq_constrs:
            exprs += [constr.args[0]]
        self.n_eq = sum([e.size for e in exprs])

        neq_constrs = constr_map[NonPos]
        for constr in neq_constrs:
            if isinstance(constr, NonPos):
                exprs += [constr.args[0]]
                self.n_leq += constr.size
            else:
                raise Exception("Unknown constraint type")

        constr_op = self.expr_handler.aff_operator(exprs, 'constraint',
                                                    x_length, var_offsets)
        self.A_sparsity = constr_op.coeff
