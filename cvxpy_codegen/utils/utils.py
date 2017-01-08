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
from jinja2 import contextfilter
from cvxpy_codegen import CallbackParam, Constant, Parameter, Variable, __path__
import os

PKG_PATH = os.path.dirname(os.path.abspath(__path__[0]))
FILE_SEP = '/' # TODO generalize for windows

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