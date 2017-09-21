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
import scipy.sparse as sp
import numpy as np


class IndexData(AffAtomData):

    def get_atom_data(self, expr, arg_data, arg_pos):

        # Define behavior for each case.
        slices = expr.get_data()[0]
        if arg_data[0].is_scalar():
            raise Exception("Indexing scalars not allowed.")

        if arg_data[0].is_vector():
            if not len(slices) == 1:
                raise Exception("Vectors cannot be multiply indexed.")
            slices = (slices[0], slice(0, 1, 1))

        if arg_data[0].is_matrix():
            if len(slices) == 1:
                slices = (slices[0], slice(0, None, 1))


        start0 = 0 if slices[0].start==None else slices[0].start
        start1 = 0 if slices[1].start==None else slices[1].start
        stop0 = arg_data[0].sparsity.shape[0] if slices[0].stop==None else slices[0].stop
        stop1 = arg_data[0].sparsity.shape[1] if slices[1].stop==None else slices[1].stop
        step0 = 1 if slices[0].step==None else slices[0].step
        step1 = 1 if slices[1].step==None else slices[1].step

        if start0 < 0 or stop0 > arg_data[0].shape[0]:
            raise ValueError("First index out of bounds")
        if start1 < 0 or stop1 > arg_data[0].shape[1]:
            raise ValueError("Second index out of bounds")

        data = {'start0' : start0,
                'stop0'  : stop0,
                'step0'  : step0,
                'start1' : start1,
                'stop1'  : stop1,
                'step1'  : step1}

        sparsity = arg_data[0].sparsity[start0 : stop0 : step0,
                                        start1 : stop1 : step1]

        return ConstExprData(expr, arg_data,
                        macro_name = 'index',
                        sparsity = sparsity,
                        data = data)



    def get_coeff_data(self, args, var):

        # Define behavior for each case.
        slices = self.get_data()[0]
        if args[0].is_scalar():
            raise Exception("Indexing scalars not allowed.")

        if args[0].is_vector():
            if not len(slices) == 1:
                raise Exception("Vectors cannot be multiply indexed.")
            slices = (slices[0], slice(0, 1, 1))

        if args[0].is_matrix():
            if len(slices) == 1:
                slices = (slices[0], slice(0, None, 1))

        sz0, sz1 = args[0].shape
        slice0 = slices[0]
        slice1 = slices[1]
        
        # Get index slice data.
        step0 = 1 if slice0.step==None else slice0.step
        step1 = 1 if slice1.step==None else slice1.step
        data = {'start0' : slice0.start,
                'stop0'  : slice0.stop,
                'step0'  : step0,
                'start1' : slice1.start,
                'stop1'  : slice1.stop,
                'step1'  : step1}

        if data['stop0'] > sz0 or data['stop1'] > sz1:
            raise Exception("Index out of bounds.") 

        idxs0 = range(*slice0.indices(sz0))
        idxs1 = range(*slice1.indices(sz1))
        indices = []
        for idx1 in idxs1:
          for idx0 in idxs0:
            indices += [idx0 + sz0 * idx1]
        sp.csr_matrix(args[0].sparsity)[indices,:]
        sparsity = args[0].sparsity[indices, :]

        return CoeffData(self, args, var,
                         sparsity = sparsity,
                         macro_name = 'index_coeffs',
                         data = data)
