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

import scipy.sparse as sp
import numpy as np
from cvxpy_codegen import CallbackParam, Constant, Parameter, Variable, __path__
import os
from jinja2 import Environment, PackageLoader, contextfilter

PKG_PATH = os.path.dirname(os.path.abspath(__path__[0]))
EXP_CONE_LENGTH = 3

class Counter():
    def __init__(self):
        self.count = 0

    def get_count(self):
        c = self.count
        self.count += 1
        return c


# Get a sparse zero matrix.
def spzeros(m, n, dtype=float):
    return sp.csr_matrix(np.zeros((m,n)), dtype=dtype)


@contextfilter
def call_macro(context, macro_name, *args , **kwargs):
    return context.vars[macro_name](*args, **kwargs)


def make_target_dir(target_dir):
    target_dir = os.path.abspath(target_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)


# General functions and classes needed to evaluate the templates. # TODO review these
DEFAULT_TEMPLATE_VARS = {'isinstance' : isinstance,
                         'len'        : len,
                         'enumerate'  : enumerate,
                         'id'         : id, 
                         'type'       : type,
                         'set'        : set,
                         'CallbackParam': CallbackParam,
                         'Constant'   : Constant,
                         'Parameter'  : Parameter,
                         'Variable'   : Variable }


def render(target_dir, template_vars, template_path, target_name):
    total_template_vars = dict()
    total_template_vars.update(template_vars)
    total_template_vars.update(DEFAULT_TEMPLATE_VARS)

    env = Environment(loader=PackageLoader('cvxpy_codegen', ''),
                      lstrip_blocks=True,
                      trim_blocks=True)
    env.filters['call_macro'] = call_macro

    template = env.get_template(template_path)
    f = open(os.path.join(target_dir, target_name), 'w')
    f.write(template.render(total_template_vars))
    f.close()
