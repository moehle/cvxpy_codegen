from cvxpy_codegen.param.expr_data import AtomData
import scipy.sparse as sp
import numpy as np

def getdata_mul(linop, arg_data):
    sp_data = sp.csr_matrix(linop.data)
    coeff = np.blkdiag([sp_data] * arg_data.size[1])
    sparsity = coeff * arg_data[0].sparsity
    return AtomData(expr, arg_data,
                    macro_name = "mul",
                    sparsity = sparsity,
                    work_int = linop.size[0] * linop.size[1],
                    work_float = linop.size[0] * linop.size[1])
