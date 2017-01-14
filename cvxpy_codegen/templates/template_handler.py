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

from cvxpy_codegen.utils.utils import render
import numpy


INTF_SOURCES = ['codegen.c',
                'linop.c',
                'param.c',
                'solver_intf.c',
                'codegenmodule.c', ]

INTF_INCLUDE_DIRS = [numpy.get_include()]


class TemplateHandler():


    def __init__(self, target_dir):
        self.target_dir = target_dir


    def get_template_vars(self):
        self.template_vars = {'intf_sources'      : INTF_SOURCES,
                              'intf_include_dirs' : INTF_INCLUDE_DIRS}
        return self.template_vars

        
    def render(self, target_dir, template_vars):
        template_vars.update(self.template_vars)
        render(target_dir, template_vars, 'templates/codegen.c.jinja', 'codegen.c')
        render(target_dir, template_vars, 'templates/codegen.h.jinja', 'codegen.h')
        render(target_dir, template_vars, 'templates/codegenmodule.c.jinja', 'codegenmodule.c')
        render(target_dir, template_vars, 'templates/cvxpy_codegen_solver.py.jinja', 'cvxpy_codegen_solver.py')
        render(target_dir, template_vars, 'templates/setup.py.jinja', 'setup.py')
        render(target_dir, template_vars, 'templates/Makefile.jinja', 'Makefile')
        render(target_dir, template_vars, 'templates/example_problem.c.jinja', 'example_problem.c')
