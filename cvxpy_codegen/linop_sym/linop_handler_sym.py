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
import cvxpy_codegen.linop_sym.lin_to_mat as l2m
import cvxpy.lin_ops as lo
import cvxpy_codegen.linop_sym.sym_matrix as sym
from cvxpy_codegen.linop_sym.sym_expr import SymAdd, SymMult, SymDiv, SymParam, SymConst
from cvxpy_codegen import Variable
from cvxpy_codegen.utils.utils import render, EXP_CONE_LENGTH
import cvxpy.settings as s



LINOP_TEMPLATE_VARS = {'SymAdd'    : SymAdd,
                       'SymMult'   : SymMult,
                       'SymDiv'    : SymDiv,
                       'SymParam'  : SymParam,
                       'SymConst'  : SymConst,
                       'Variable'  : Variable }

class LinOpHandlerSym():


    def __init__(self, sym_data, obj, eq_constr, leq_constr):
        self.sym_data = sym_data
        self.obj = obj
        self.eq_constr = eq_constr

        #print("\nWEM")
        #print(eq_constr)

        self.leq_constr = leq_constr
        self.template_vars = LINOP_TEMPLATE_VARS


    def obj_to_mat(self, obj):
        obj_coeff, obj_offset = self.process_linop(obj, 
                                    horz_size=self.sym_data.x_length)
        return obj_coeff, obj_offset


    def constrs_to_mat(self, constraints):
        constr_size = sum([c.size[0] * c.size[1] for c in constraints])
        #print("constraint sizes:",  [c.size[0] for c in constraints])
        #print("linop sizes:",  [c.expr.size[0]*c.expr.size[1] for c in constraints])
        #print("constraint size: %d" % constr_size)
        x_len = self.sym_data.x_length
        matrix = sym.zeros(constr_size, x_len)
        offset = sym.zeros(constr_size, 1)
        vert_offset = 0

        for constr in constraints:
            c_mat, c_off = self.process_linop(constr.expr, x_len,
                               vert_offset=vert_offset, vert_size=constr_size)
            matrix += c_mat
            offset += c_off
            vert_offset += constr.size[0] * constr.size[1]
        return matrix, -offset


    def get_template_vars(self):
        eq_coeff, eq_offset = self.constrs_to_mat(self.eq_constr) # TODO change order
        obj_coeff, obj_offset = self.obj_to_mat(self.obj)
        leq_coeff, leq_offset = self.constrs_to_mat(self.leq_constr)

        #self.sprs_data.update_eq_constr(eq_coeff)
        #self.sprs_data.update_leq_constr(leq_coeff)

        #print("\n\nEQ_COEFF")
        #print(eq_coeff.Ap)
        #print(eq_coeff.Ai)
        #print(eq_coeff.Ax)
        #print(eq_coeff.m)
        #print(eq_coeff.n)
        #print("\n\nEQ_OFFSET")
        #print(eq_offset.Ap)
        #print(eq_offset.Ai)
        #print(eq_offset.Ax)
        #print(eq_offset.m)
        #print(eq_offset.n)
        #print(eq_offset.Ax[0].args)

        #print("\n\nTHIS")
        #print(self.named_vars)
        #print(self.named_vars)

        #self.template_vars["callback_params"] = other_tvs["callback_params"] # TODO remove dependency on param_handler


        # TODO this is duplicated from ecos_intf.py
        dims = self.sym_data.dims
        cone_dim = (dims[s.LEQ_DIM] + 
                    EXP_CONE_LENGTH*dims[s.EXP_DIM] + 
                    sum(dims[s.SOC_DIM]))

        self.template_vars.update({
                'x_length'     :    self.sym_data.x_length,
                'leq_dim'      :    dims[s.LEQ_DIM],
                'eq_dim'       :    dims[s.EQ_DIM],
                'cone_dim'     :    cone_dim })
        # TODO are leq_dim and cone_dim the same?
        
        self.template_vars.update({'obj_coeff'  : obj_coeff,
                                   'obj_offset' : obj_offset,
                                   'eq_coeff'   : eq_coeff,
                                   'eq_offset'  : eq_offset,
                                   'leq_coeff'  : leq_coeff,
                                   'leq_offset' : leq_offset })

        #print('BLAH')
        #print(self.template_vars['eq_coeff'].nnz)

        return self.template_vars


    def process_linop(self, linop, horz_size, vert_offset=0, vert_size=1):
        ## Get the variables.
        #self.get_var_names(linop)

        # Get the coefficients.
        coeffs = l2m.get_coefficients(linop) 
        #print('\nCOEFFS')
        #print(coeffs[0][1].Ap)
        #print(coeffs[0][1].Ai)
        #print(coeffs[0][1].Ax)
        #print(coeffs[0][1].nnz)
        #print(coeffs[1][1].Ap)
        #print(coeffs[1][1].Ai)
        #print(coeffs[1][1].Ax)
        #print(coeffs[1][1].nnz)
        offset = sym.zeros(vert_size, 1)
        matrix = sym.zeros(vert_size, horz_size)
        for id_, sym_mat in coeffs:
            vert_start = vert_offset 
            vert_end = vert_start + linop.size[0]*linop.size[1] 
            if id_ is lo.CONSTANT_ID: 
                sym_mat = sym.zero_pad(sym_mat, (vert_size, 1),
                                       (vert_start, 0))
                offset += sym_mat
            else: 
                horiz_offset = self.sym_data.var_offsets[id_] 
                #print("\n")
                #print(linop.type)
                #print(vert_size)
                #print(horz_size)
                #print(vert_start)
                #print(horiz_offset)
                #print(sym_mat.size)
                #print(linop.size)
                sym_mat = sym.zero_pad(sym_mat, (vert_size, horz_size),
                                       (vert_start, horiz_offset))

                #print('\nmatrix:')
                #print(matrix.Ap)
                #print(matrix.Ai)
                #print(matrix.Ax)
                #print(matrix.nnz)
                #print(matrix.size)

                #print('\nsym_mat')
                #print(sym_mat.Ap)
                #print(sym_mat.Ai)
                #print(sym_mat.Ax)
                #print(sym_mat.nnz)
                #print(sym_mat.size)
                matrix += sym_mat
                #print('\nmatrix:')
                #print((matrix + sym_mat).Ap)
                #print((matrix + sym_mat).Ai)
                #print((matrix + sym_mat).Ax)
                #print((matrix + sym_mat).nnz)
                #print((matrix + sym_mat).size)

        #print('\n')
        #print(matrix.Ap)
        #print(matrix.Ai)
        #print(matrix.Ax)
        #print(matrix.nnz)
        #print(matrix.size)

        return matrix, offset



    def render(self, target_dir):
        render(target_dir, self.template_vars,
               'linop_sym/linop_sym.c.jinja', 'linop.c')


#TODO clean up this file
