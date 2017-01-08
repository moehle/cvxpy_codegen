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
from cvxpy_codegen.utils.utils import FILE_SEP, call_macro, DEFAULT_TEMPLATE_VARS
from cvxpy_codegen.solvers.embedded_solver_intf import EmbeddedSolverIntf
import os
import shutil as sh
import cvxpy.settings as s
from cvxpy_codegen.utils.utils import FILE_SEP, PKG_PATH
from glob import glob
import ecos
import cvxpy.problems.solvers.utilities as cvxpy_utils

CVXPY_ECOS = cvxpy_utils.SOLVERS["ECOS"]
EXP_CONE_LENGTH = 3

class EcosIntf(EmbeddedSolverIntf):

    name = 'ecos'
    REQ_INIT = True
    REQ_CLEANUP = True
    SOLVER_TEMPLATE = "solvers/ecos_intf.c.jinja"

    SOURCE_DIR = PKG_PATH + FILE_SEP + 'cvxpy_codegen/solvers/ecos'
    IGNORES = None

    # How should CVXPY should handle get sym_data for this embedded solver:
    CVXPY_SOLVER = CVXPY_ECOS


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
                      'solver_template'         : SOLVER_TEMPLATE } 


    def get_template_vars(self, sym_data, other_tvs):

        dims = sym_data.dims
        cone_dim = (dims[s.LEQ_DIM] + 
                    EXP_CONE_LENGTH*dims[s.EXP_DIM] + 
                    sum(dims[s.SOC_DIM])) # TODO

        self.template_vars.update(DEFAULT_TEMPLATE_VARS)

        self.template_vars.update({
                'leq_dim'      :    dims[s.LEQ_DIM],
                'eq_dim'       :    dims[s.EQ_DIM],
                'soc_dims'     :    dims[s.SOC_DIM],
                'exp_cones'    :    dims[s.EXP_DIM],
                'x_length'     :    sym_data.x_length,
                'cone_dim'     :    cone_dim })

        self.template_vars['leq_nnz'] = other_tvs['leq_coeff'].nnz
        self.template_vars['eq_nnz']  = other_tvs['eq_coeff'].nnz

        return self.template_vars


    # ECOS doesn't really have code gen, so just copy all the source files:
    def render(self, target_dir):
        solver_dir = target_dir + FILE_SEP + self.name
        
        # make target directory
        if os.path.exists(solver_dir):
            sh.rmtree(solver_dir)
        
        # Copy ecos source files
        sh.copytree(self.SOURCE_DIR, solver_dir, ignore=self.IGNORES)

        # Render interface template:
        env = Environment(loader=PackageLoader('cvxpy_codegen', ''),
                          lstrip_blocks=True,
                          trim_blocks=True)
        env.filters['call_macro'] = call_macro
        solver_intf_c = env.get_template('solvers/ecos_intf.c.jinja')

        # render and save source file # TODO
        s = target_dir + FILE_SEP + 'solver_intf.c'
        f = open(target_dir + FILE_SEP + 'solver_intf.c', 'w')
        f.write(solver_intf_c.render(self.template_vars))
        f.close()
