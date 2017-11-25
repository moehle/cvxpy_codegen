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

from cvxpy_codegen.object_data.expr_data import ExprData


class CoeffData(ExprData):
    def __init__(self, atom_data, arg_data, vid,
                 sparsity=None,
                 inplace=False,
                 macro_name=None,
                 work_int=0,
                 work_float=0,
                 work_coeffs=0,
                 data=None,
                 copy_arg=0):
        ExprData.__init__(self, atom_data, arg_data, sparsity=sparsity)
        self.inplace = inplace
        self.macro_name = macro_name
        self.work_int = work_int
        self.work_float = work_float
        self.work_coeffs = work_coeffs
        self.data = data
        self.name = atom_data.name + '_var' + str(vid)
        self.type = 'coeff'
        self.vid = vid
        self.var_ids = [vid]
        self.shape = atom_data.shape
        self.copy_arg = copy_arg
        self.atom_data = atom_data
        has_const_or_param = True # TODO change this, wrong
        #has_const_or_param = any([a.overwriteable for a in arg_data])
        #has_const_or_param = any([isinstance(a, ConstData) or
        #                          isinstance(a, ExprData) or
        #                          isinstance(a, VarData) for a in arg_data])
        if inplace and has_const_or_param:
            self.make_copy = True
        else:
            self.make_copy = False
        self.data = data
        self.name = self.storage.name # TODO rm?
        self.c_name = "&work->%s" % self.storage.name



    def get_coeffs(self):
        return dict()

    def get_offset_expr(self):
        return self



    @property
    def storage(self):
        if self.inplace and not self.make_copy:
            return self.args[self.copy_arg].storage
        else:
            return self

    def force_copy(self):
        self.make_copy = True


    def codegen(self):
        s = ""
        if self.make_copy:
            s += "copy_linop(%s, %s);\n" % (self.args[self.copy_arg].c_name, self.c_name)
        return s + self.atom_data.codegen_coeff(self)
