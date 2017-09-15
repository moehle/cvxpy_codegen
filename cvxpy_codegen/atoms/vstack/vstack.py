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

from cvxpy_codegen.object_data.atom_data import AtomData
from cvxpy_codegen.object_data.linop_coeff_data import LinOpCoeffData
import scipy.sparse as sp

def atomdata_vstack(expr, arg_data, arg_pos):
    offsets = []
    vert_offset = 0
    for i, a in enumerate(expr.args):
        if i in arg_pos:
            offsets += [vert_offset]
        vert_offset += a.shape[0]

    work_varargs = len(arg_data) # This is a varargs atom.
    work_int = len(arg_data)

    ndims = len(expr.shape)
    if ndims == 2:
        shape = expr.shape
    elif ndims == 1:
        shape = (expr.shape[0], 1)
    elif ndims == 0:
        shape = (1,1)
    else:
        raise Exception("Code generation only supports arrays"
                        "with two or fewer dimensions.")

    sparsity = sp.lil_matrix(shape, dtype='bool')
    for a, o in zip(arg_data, offsets):
        m = a.shape[0]
        sparsity[o:o+m, :] = a.sparsity

    #sparsity = sp.vstack([a.sparsity for a in arg_data])
    return AtomData(expr, arg_data,
                    macro_name = "vstack",
                    sparsity = sp.csr_matrix(sparsity),
                    work_int = work_int,
                    data = offsets,
                    work_varargs = work_varargs)





def coeffdata_vstack(linop, args, var):
    # TODO replace with arg_pos:
    vert_offset = 0
    offsets = []
    for a in linop.args:
        if var in a.var_ids:
            offsets += [vert_offset]
        vert_offset += a.shape[0]

    m_var = linop.shape[0]
    n_var = args[0].shape[1]
    n = args[0].sparsity.shape[1]
    sparsity = sp.lil_matrix((m_var*n_var, n), dtype='bool')
    for j in range(n_var):
        mats = []
        for a, o in zip(args, offsets):
            m = a.shape[0]
            #mats += [a.sparsity[j*m : (j+1)*m, :]]
            #print(m_var)
            #print(j)
            #print(o)
            #print(m)
            #print(sparsity[m_var*j+o:m_var*j+o+m,:].shape)
            #print(a.sparsity[j*m : (j+1)*m, :].shape)
            sparsity[m_var*j+o:m_var*j+o+m,:] = a.sparsity[j*m : (j+1)*m, :]
        #sparsity = sp.vstack([sparsity] + mats)

    work_coeffs = len(args) # This is a varargs linop.
    work_int = len(args)
    return LinOpCoeffData(linop, args, var,
                          sparsity = sp.csr_matrix(sparsity),
                          work_int = work_int,
                          work_coeffs = work_coeffs,
                          data = offsets,
                          macro_name = 'vstack_coeffs')
