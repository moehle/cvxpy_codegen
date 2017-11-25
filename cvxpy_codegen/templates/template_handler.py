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
                'expr_handler.c',
                'solver_intf.c',
                'constants.h',
                'atoms.h',
                'atoms.c',
                'codegenmodule.c', ]

RENDER_TARGETS = {
        'templates/atoms.c'                       : 'atoms.c',
        'templates/atoms.h'                       : 'atoms.h',
        'templates/codegen.c.jinja'               : 'codegen.c',
        'templates/codegen.h.jinja'               : 'codegen.h',
        'templates/codegenmodule.c.jinja'         : 'codegenmodule.c',
        'templates/cvxpy_codegen_solver.py.jinja' : 'cvxpy_codegen_solver.py',
        'templates/setup.py.jinja'                : 'setup.py',
        'templates/Makefile.jinja'                : 'Makefile',
        'templates/example_problem.c.jinja'       : 'example_problem.c',
        'templates/constants.h.jinja'             : 'constants.h' }


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
        for template in RENDER_TARGETS:
            target = RENDER_TARGETS[template]
            render(target_dir, template_vars, template, target)
