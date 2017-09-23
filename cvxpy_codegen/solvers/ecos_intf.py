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
from cvxpy import vstack, reshape


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



    def get_template_vars(self):

        # Recover the sparsity patterns of the coefficient matrices.
        #x_length, var_offsets = self.expr_handler.get_sym_data()
        x_length = self.inv_data.x_length
        var_offsets = self.inv_data.var_offsets
        obj_coeff, eq_coeff, leq_coeff = self.get_sparsity(x_length, var_offsets)

        # TODO rename many of these:
        self.template_vars.update({
                'obj_coeff'    : obj_coeff,
                'eq_coeff'     : eq_coeff,
                'leq_coeff'    : leq_coeff,
                'objective'    : self.obj,
                'eq_constr'    : self.eq_constrs,
                'leq_constr'   : self.leq_constrs,
                'leq_dim'      : self.n_leq,
                'eq_dim'       : self.n_eq,
                'soc_dims'     : self.soc_sizes,
                'exp_cones'    : self.n_exp,
                'cone_dim'     : self.cone_dim,
                'leq_nnz'      : self.leq_nnz,
                'eq_nnz'       : self.eq_nnz,
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
        render(target_dir, self.template_vars, 'solvers/ecos_intf.c.jinja', 'solver_intf.c')


    def process_problem(self, obj, constrs):

        # TODO reorganize?
        #self.obj = obj
        #self.constrs = constrs

        # Separate constraints.
        constr_map = group_constraints(constrs)

        # Process objective.
        self.obj = self.expr_handler.process_expression(obj)

        # Process equality constraints.
        vert_offset = 0
        eq_constrs = constr_map[Zero]
        for constr in eq_constrs:
            exprs = []
            for a in constr.args:
                exprs += [self.expr_handler.process_expression(a)]
            self.eq_constrs += [ZeroData(constr, exprs, vert_offset=vert_offset)]
            vert_offset += constr.size
        self.n_eq = vert_offset

        # Process conic constraints.
        neq_constrs = constr_map[NonPos] + constr_map[SOC] + constr_map[ExpCone]
        vert_offset = 0
        for constr in neq_constrs:
            if isinstance(constr, NonPos):
                expr = self.expr_handler.process_expression(constr.args[0])
                self.leq_constrs += [NonPosData(constr, [expr], 
                                     vert_offset=vert_offset)]
                self.n_leq += constr.size
                vert_offset += constr.size
            elif isinstance(constr, SOC):
                expr = self.get_soc_expr(constr)
                expr = self.expr_handler.process_expression(expr)
                self.leq_constrs += [SocData(constr, [expr],
                                     vert_offset=vert_offset)]
                self.n_soc += len(constr.cone_sizes())
                self.soc_sizes += constr.cone_sizes()
                vert_offset += sum(constr.cone_sizes())
            elif isinstance(constr, ExpCone):
                raise Exception("Exp cone not implemented")
                exprs = []
                for a in constr.args:
                    exprs += [self.expr_handler.process_expression(a)]
                self.leq_constrs += [ExpConeData(constr, exprs,
                                     vert_offset=vert_offset)]
                self.n_exp += constr.sizesize
                vert_offset += constr.size
            else:
                raise Exception("Unknown constraint type")

        self.cone_dim = self.n_leq + 3*self.n_exp + sum(self.soc_sizes)



    # For each constraint, collect data on the constraint,
    # then process its expression trees.
    def process_constr(self, constrs):
        constr_data = []
        vert_offset = 0
        return constr_data
    

    def get_soc_expr(self, c):
        T = c.args[0]
        X = c.args[1]
        if c.axis == 1:
            X = X.T
        if len(X.shape) == 0:
            reshape(X, (1, 1))
        elif len(X.shape) == 1:
            X = reshape(X, (X.shape[0], 1))
        n, n_cones = X.shape
        T = reshape(T, (1, n_cones))
        return -reshape(vstack([T,X]), ((n+1)*n_cones,))
 




    # Gets the sparsity patterns of the objective and constraint coefficients.
    # (This tells us how much memory to allocate in C).
    def get_sparsity(self, x_length, var_offsets):

        # Get Boolean sparse matrix for the objective.
        obj_coeff = self.obj.get_matrix(x_length, var_offsets) 
        
        # Get Boolean sparse matrix for the equality constraints.
        eq_coeff = sp.csc_matrix((0, x_length), dtype=bool)
        for c in self.eq_constrs:
            eq_coeff = sp.vstack([eq_coeff,
                    c.get_matrix(x_length, var_offsets)])

        # Get Boolean sparse matrix for the inequality constraints.
        leq_coeff = sp.csc_matrix((0, x_length), dtype=bool)
        for c in self.leq_constrs:
            leq_coeff = sp.vstack([leq_coeff,
                    c.get_matrix(x_length, var_offsets)])

        self.leq_nnz = leq_coeff.nnz
        self.eq_nnz = eq_coeff.nnz

        return (sp.csc_matrix(obj_coeff),
                sp.csc_matrix(eq_coeff),
                sp.csc_matrix(leq_coeff))
