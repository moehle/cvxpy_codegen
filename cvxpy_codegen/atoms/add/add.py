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

from cvxpy_codegen.object_data.const_expr_data import ConstExprData
from cvxpy_codegen.object_data.coeff_data import CoeffData
from cvxpy_codegen.object_data.aff_atom_data import AffAtomData


class AddData(AffAtomData):

    def get_atom_data(self, expr, arg_data, arg_pos):
    
        if len(arg_data) == 1 and arg_data[0].shape != (1,1):
            return ConstExprData(self, arg_data,
                                 inplace = True,
                                 sparsity = arg_data[0].sparsity,
                                 macro_name = 'null')
    
        else:
            if any([a.sparsity.shape==(1,1) for a in arg_data]):
                sparsity = None
                work_int    = max([a.shape[1] for a in arg_data])
                work_float  = max([a.shape[1] for a in arg_data])
            else:
                sparsity = sum([a.sparsity for a in arg_data])
                work_int    = max([a.shape[1] for a in arg_data])
                work_float  = max([a.shape[1] for a in arg_data])
            work_varargs    = len(arg_data) # This is a varargs atom.
            return ConstExprData(self, arg_data,
                                 sparsity = sparsity,
                                 work_int = work_int,
                                 work_float = work_float,
                                 work_varargs = work_varargs,
                                 macro_name = 'add')


    def codegen_offset(self, expr):
        if len(expr.args) == 1 and expr.args[0].shape != (1,1):
            return ""
        else:
            s = ""
            for i, c in enumerate(expr.args):
                s += "work->work_varargs[%d] = work->%s;\n" % (i, c.storage.name)
            s += "add(%d, work->work_varargs, %s, work->work_int, work->work_double);\n" \
                        % (len(expr.args), expr.c_name)
            return s
        


    def get_coeff_data(self, args, var):
        if len(args) == 1 and args[0].shape == self.shape:
            return CoeffData(self, args, var,
                             inplace = True,
                             sparsity = args[0].sparsity,
                             macro_name = 'null')
    
        else:
            work_coeffs = len(args) # This is a varargs expr.
            sparsity = sum([a.sparsity for a in args])
            work_int    = sparsity.shape[1]
            work_float  = sparsity.shape[1]
            return CoeffData(self, args, var,
                             sparsity = sparsity,
                             work_int = work_int,
                             work_float = work_float,
                             work_coeffs = work_coeffs,
                             macro_name = 'add_coeffs')


    def codegen_coeff(self, expr):
        if len(expr.args) == 1 and expr.args[0].shape == self.shape:
            return ""
        else:
            s = ""
            for i, c in enumerate(expr.args):
                s += "work->work_coeffs[%d] = work->%s;\n" % (i, c.storage.name)
            s += "add_coeffs(%d, work->work_coeffs, %s, work->work_int, work->work_double);\n" \
                        % (len(expr.args), expr.c_name)
            return s
