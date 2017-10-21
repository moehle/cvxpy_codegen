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
from cvxpy_codegen.object_data.constr_data import \
        NonPosData, SocData, ExpConeData, ZeroData
import scipy.sparse as sp
from cvxpy import vstack, reshape

from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.matrix_stuffing import MatrixStuffing



class EcosIntf(EmbeddedSolverIntf):
    name = 'ecos'
    SOLVER_TEMPLATE = "solvers/ecos_intf.c.jinja"

    SOURCE_DIR = os.path.join(PKG_PATH, 'cvxpy_codegen/solvers/ecos')
    IGNORES = None

    # Macros to be used for the Python interface.
    DEFINE_MACROS = [('PYTHON',None),
                     ('DLONG', None),
                     ('LDL_LONG', None),
                     ('CTRLC', 1)]

    INCLUDE_DIRS = ['ecos/include',
                    'ecos/external/amd/include',
                    'ecos/external/ldl/include',
                    'ecos/external/SuiteSparse_config']

    SOURCES = (['ecos/external/ldl/src/ldl.c',
                'ecos/src/cone.c',
                'ecos/src/ctrlc.c',
                'ecos/src/ecos.c',
                'ecos/src/equil.c',
                'ecos/src/expcone.c',
                'ecos/src/kkt.c',
                'ecos/src/preproc.c',
                'ecos/src/spla.c',
                'ecos/src/splamm.c',
                'ecos/src/timer.c',
                'ecos/src/wright_omega.c',
                'ecos/external/amd/src/amd_1.c',
                'ecos/external/amd/src/amd_aat.c',
                'ecos/external/amd/src/amd_defaults.c',
                'ecos/external/amd/src/amd_global.c',
                'ecos/external/amd/src/amd_order.c',
                'ecos/external/amd/src/amd_post_tree.c',
                'ecos/external/amd/src/amd_valid.c',
                'ecos/external/amd/src/amd_2.c',
                'ecos/external/amd/src/amd_control.c',
                'ecos/external/amd/src/amd_dump.c',
                'ecos/external/amd/src/amd_info.c',
                'ecos/external/amd/src/amd_postorder.c',
                'ecos/external/amd/src/amd_preprocess.c'])

    template_vars = { 'solver_sources'          : SOURCES,
                      'solver_include_dirs'     : INCLUDE_DIRS,
                      'solver_define_macros'    : DEFINE_MACROS,
                      'solver_template'         : SOLVER_TEMPLATE,
                      'solver_name'             : name } 

    
    def __init__(self, expr_handler, inv_data, include_solver=True):
        self.expr_handler = expr_handler
        self.eq_constrs = []
        self.leq_constrs = []
        self.inv_data = inv_data
        self.n_eq = 0
        self.n_leq = 0
        self.n_soc = 0
        self.n_exp = 0
        self.soc_sizes = []
        self.cone_dim = 0
        self.template_vars.update({'include_solver' : include_solver})

                
    # TODO untested:
    def preprocess_problem(self, prob):
        sc = construct_solving_chain(prob, solver="ECOS")

        for r in sc.reductions:
            if isinstance(r, Dcp2Cone):
                prob, dcp2cone_inv_data = r.apply(prob)
            if isinstance(r, CvxAttr2Constr):
                prob, attr_inv_data = r.apply(prob)

        if not attr_inv_data == ():
            id2new_var, id2old_var, __ = attr_inv_data
            for id in id2old_var.keys():
                # TODO Sets a private attribute :(
                id2new_var[id]._name = id2old_var[id].name()

        return prob


    def process_problem(self, prob):
        obj = prob.objective.expr
        constrs = prob.constraints

        # Separate constraints.
        constr_map = group_constraints(constrs)
        x_length = self.inv_data.x_length
        var_offsets = self.inv_data.var_offsets

        self.expr_handler.aff_functional(
                obj, 'obj', x_length, var_offsets)

        exprs = []
        eq_constrs = constr_map[Zero]
        for constr in eq_constrs:
            exprs += [constr.args[0]]
        self.expr_handler.aff_operator(exprs, 'eq', x_length, var_offsets)
        self.n_eq = sum([e.size for e in exprs])

        neq_constrs = constr_map[NonPos] + constr_map[SOC] + constr_map[ExpCone]
        exprs = []
        for constr in neq_constrs:
            if isinstance(constr, NonPos):
                exprs += [constr.args[0]]
                self.n_leq += constr.size
            elif isinstance(constr, SOC):
                exprs += [self._get_soc_expr(constr)]
                self.n_soc += len(constr.cone_sizes())
                self.soc_sizes += constr.cone_sizes()
            elif isinstance(constr, ExpCone):
                raise Exception("Exp cone not implemented")
            else:
                raise Exception("Unknown constraint type")
        self.expr_handler.aff_operator(exprs, 'leq', x_length, var_offsets)

        self.cone_dim = self.n_leq + 3*self.n_exp + sum(self.soc_sizes)



    def get_template_vars(self):

        # Recover the sparsity patterns of the coefficient matrices.
        x_length = self.inv_data.x_length
        var_offsets = self.inv_data.var_offsets

        # TODO rename many of these:
        self.template_vars.update({
                'leq_dim'      : self.n_leq,
                'eq_dim'       : self.n_eq,
                'soc_dims'     : self.soc_sizes,
                'exp_cones'    : self.n_exp,
                'cone_dim'     : self.cone_dim,
                'var_offsets'  : var_offsets,
                'x_length'     : x_length})

        return self.template_vars


    # ECOS doesn't really have code gen, so just copy all the source files:
    def render(self, target_dir):
        solver_dir = os.path.join(target_dir, self.name)
        
        # make target directory
        if os.path.exists(solver_dir):
            sh.rmtree(solver_dir)
        
        # Copy ecos source files
        sh.copytree(self.SOURCE_DIR, solver_dir, ignore=self.IGNORES)

        # Render interface template:
        render(target_dir, self.template_vars,
               'solvers/ecos_intf.c.jinja', 'solver_intf.c')


    def _get_soc_expr(self, c):
        T = c.args[0]
        X = c.args[1]
        c.axis = 0
        if c.axis == 1:
            X = X.T
        if len(X.shape) == 0:
            reshape(X, (1, 1))
        elif len(X.shape) == 1:
            X = reshape(X, (X.shape[0], 1))
        n, n_cones = X.shape
        T = reshape(T, (1, n_cones))
        return -reshape(vstack([T,X]), ((n+1)*n_cones,))
