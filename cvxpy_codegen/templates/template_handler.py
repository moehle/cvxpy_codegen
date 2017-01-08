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
from cvxpy_codegen import CallbackParam, Constant, Parameter, Variable
import os
from cvxpy_codegen.utils.utils import FILE_SEP, PKG_PATH
import numpy


INTF_SOURCES = ['codegen.c',
                'linop.c',
                'param.c',
                'solver_intf.c',
                'codegenmodule.c', ]

INTF_INCLUDE_DIRS = [numpy.get_include()]

LIB = PKG_PATH + FILE_SEP + 'lib'


# General functions and classes needed to evaluate the templates. # TODO review these
DEFAULT_TEMPLATE_VARS = {'isinstance' :  isinstance,
                         'len'        :  len,
                         'enumerate'  :  enumerate,
                         'id': id, 
                         'type': type,
                         'set': set,
                         'CallbackParam': CallbackParam,
                         'Constant': Constant,
                         'Parameter': Parameter,
                         'Variable': Variable }


@contextfilter
def call_macro(context, macro_name, *args , **kwargs):
    return context.vars[macro_name](*args, **kwargs)



class TemplateHandler():


    def __init__(self, target_dir):
        self.target_dir = target_dir
        self.env = Environment(loader=PackageLoader('cvxpy_codegen', ''),
                               lstrip_blocks=True,
                               trim_blocks=True)
        self.env.filters['call_macro'] = call_macro
        self.templates = []



    def get_template_vars(self):
        template_vars = dict()
        template_vars.update(DEFAULT_TEMPLATE_VARS)
        template_vars.update({'intf_sources'      : INTF_SOURCES,
                              'intf_include_dirs' : INTF_INCLUDE_DIRS})
        return template_vars

        
    def render_and_save(self, template_vars):
        codegen_h = self.env.get_template('templates/codegen.h.jinja')
        codegen_c = self.env.get_template('templates/codegen.c.jinja')
        codegenmodule_c = self.env.get_template('templates/codegenmodule.c.jinja')
        cvxpy_codegen_solver_py = self.env.get_template('templates/cvxpy_codegen_solver.py.jinja')
        setup_py = self.env.get_template('templates/setup.py.jinja')
        makefile = self.env.get_template('templates/Makefile.jinja')
        test_solver_c = self.env.get_template('templates/test_solver.c.jinja')


        #self.render(codegen_h, 'codegen.h')
        #self.render(codegen_c, 'codegen.h')
        #self.render(codegenmodule_c, 'codegen.h')
        #self.render(cvxpy_codegen_solver_py, 'codegen.h')
        #self.render(setup_py, 'codegen.h')
        #self.render(makefile, 'codegen.h')
        #self.render(, 'codegen.h')

        # render and save include file # TODO repair template file
        f = open(self.target_dir + FILE_SEP + 'codegen.h', 'w')
        f.write(codegen_h.render(template_vars))
        f.close()
        
        # render and save source file # TODO
        f = open(self.target_dir + FILE_SEP + 'codegen.c', 'w')
        f.write(codegen_c.render(template_vars))
        f.close()

        # render and save source file # TODO
        f = open(self.target_dir + FILE_SEP + 'test_solver.c', 'w')
        f.write(test_solver_c.render(template_vars))
        f.close()

        # render and save source file # TODO
        f = open(self.target_dir + FILE_SEP + 'codegenmodule.c', 'w')
        f.write(codegenmodule_c.render(template_vars))
        f.close()

        # render and save source file # TODO
        f = open(self.target_dir + FILE_SEP + 'cvxpy_codegen_solver.py', 'w')
        f.write(cvxpy_codegen_solver_py.render(template_vars))
        f.close()

        # render and save source file # TODO
        f = open(self.target_dir + FILE_SEP + 'setup.py', 'w')
        f.write(setup_py.render(template_vars))
        f.close()

        # render and save makefile # TODO
        f = open(self.target_dir + FILE_SEP + 'Makefile', 'w')
        f.write(makefile.render(template_vars))
        f.close()


    @staticmethod
    def render(template, name):
        f = open(self.target_dir + FILE_SEP + name, 'w')
        f.write(template.render(template_vars))
        f.close()
