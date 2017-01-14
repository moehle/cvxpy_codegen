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

from cvxpy_codegen.param.expr_data import AtomData
import scipy.sparse as sp
import numpy as np

def getdata_index(expr, arg_data):

    slices = expr.get_data()[0]
    start0 = 0 if slices[0].start==None else slices[0].start
    start1 = 0 if slices[1].start==None else slices[1].start
    stop0 = arg_data[0].sparsity.shape[0] if slices[0].stop==None else slices[0].stop
    stop1 = arg_data[0].sparsity.shape[1] if slices[1].stop==None else slices[1].stop
    step0 = 1 if slices[0].step==None else slices[0].step
    step1 = 1 if slices[1].step==None else slices[1].step

    if start0 < 0 or stop0 > arg_data[0].size[0]:
        raise ValueError("First index out of bounds")
    if start1 < 0 or stop1 > arg_data[0].size[1]:
        print('\n arg size:', arg_data[0].size[1])
        print('\n stop:', stop1)
        print('\n start', start1)
        raise ValueError("Second index out of bounds")

    data = {'start0' : start0,
            'stop0'  : stop0,
            'step0'  : step0,
            'start1' : start1,
            'stop1'  : stop1,
            'step1'  : step1}

    sparsity = arg_data[0].sparsity[start0 : stop0 : step0,
                                    start1 : stop1 : step1]

    return [AtomData(expr, arg_data,
                     macro_name = 'index',
                     sparsity = sparsity,
                     data = data)]
