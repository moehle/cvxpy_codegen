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

from cvxpy.problems.problem import Problem
from cvxpy_codegen.code_generator import CodeGenerator


class Problem(Problem):


    def codegen(self, target_dir):
        vars = self.variables()
        params = self.parameters()
        obj, constraints = self.canonical_form
        CodeGenerator(obj, constraints, vars, params).codegen(target_dir)


    def __neg__(self):
        return NotImplemented
    def __add__(self, other):
        return NotImplemented
    def __radd__(self, other):
        return NotImplemented
    def __sub__(self, other):
        return NotImplemented
    def __rsub__(self, other):
        return NotImplemented
    def __mul__(self, other):
        return NotImplemented
    def __div__(self, other):
        return NotImplemented
    def __truediv__(self, other):
        return NotImplemented
