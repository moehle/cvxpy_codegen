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
from cvxpy_codegen import Parameter, CallbackParam
import numpy as np
from cvxpy_codegen.linop_sym.sym_expr import SymConst, SymParam, SymExpr

class SymMatrix():

    # Arithmetic operations with Numpy arrays should be handled by us, not Numpy:
    __array_priority__ = 100

    def __init__(self, Ap, Ai, Ax, m, n):
        if (not isinstance(Ap, list) or 
            not isinstance(Ai, list) or 
            not isinstance(Ax, list)):
            raise ValueError("Ap, Ai, and Ax must be lists")
        if not Ap[0] == 0:
            raise ValueError("Invalid format: Ap[0] must be 0, but instead is %d" % Ap[0])
        if not len(Ap) == n+1:
            raise ValueError("Invalid format (len(Ap) must be n+1)")
        if not (Ap[-1] == len(Ai) and len(Ai) == len(Ax)):
            raise ValueError("Invalid format: Ap[-1] = %d, len(Ai) = %d, len(Ax) = %d, but should be equal" % (Ap[-1], len(Ai), len(Ax)))
        if not all([isinstance(Ax[p], SymExpr) for p in range(len(Ax))]):
            raise ValueError("Invalid format: Ax contains non-SymExpr objects")
        if Ai != []:
            if not max(Ai) < m:
                raise ValueError("Invalid format (max(Ai) must be less than m)")
        self.Ap = Ap
        self.Ai = Ai
        self.Ax = Ax
        self.m = m
        self.n = n
    
    @property
    def shape(self):
        return (self.m,self.n)

    @property
    def size(self):
        return (self.m,self.n)

    @property
    def indices(self):
        return self.Ai

    @property
    def indptr(self):
        return self.Ap

    @property
    def value(self):
        Ax_vals = []
        for elem in self.Ax:
            Ax_vals += [elem.value]
        return sp.csc_matrix((Ax_vals, self.Ai, self.Ap),
                          shape=(self.m, self.n))

    def __iter__(self):
        return SymMatrixIter(self)

    @property
    def nnz(self):
        return self.Ap[-1]


    def print_Ap(self, ):
        return NotImplemented

    def print_Ai(self):
        #for i in self.Ai:
        #   s += "work->eq_nzval[%d] = %d;", % (i, );
        return NotImplemented

    def print_Ax(self, prefix = ""):
        s = ""
        for p, elem in enumerate(self.Ax):
           s += '%s[%d] = %s;\n' % (prefix, p, elem.print_self());
        return s


    def __mul__(self, arg):
        arg_scalar = get_scalar(arg)
        self_scalar = get_scalar(self)
        if arg_scalar:
            return self.mul_elem(arg_scalar)
        elif self_scalar:
            return as_sym_matrix(arg).mul_elem(self_scalar)
        else:
            return self.mul(as_sym_matrix(arg))


    def mul_elem(self, b):
        if isinstance(b, float) or isinstance(b, int):
            b = SymConst(b)
        Ax = []
        for p in range(len(self.Ax)):
            Ax += [b * self.Ax[p]]
        return SymMatrix(self.Ap, self.Ai, Ax, self.m, self.n)


    def mul(self, B):
        if (self.n != B.m):
            raise ValueError("Inner dimensions don't match.")

        Bp = B.Ap
        Bi = B.Ai
        Bx = B.Ax
        Ap = self.Ap
        Ai = self.Ai
        Ax = self.Ax
        Am = self.m
        An = self.n
        Bm = B.m
        Bn = B.n

        w = np.zeros(self.m, dtype=int)
        x = [0] * self.m
        Zp = np.zeros(B.n+1, dtype=int)
        Zi = []
        Zx = []
        for jB in range(B.n):
            Zp[jB+1] = Zp[jB]
            for pB in range(B.Ap[jB], B.Ap[jB+1]):
                iB = B.Ai[pB]
                jA = iB
                for pA in range(self.Ap[jA], self.Ap[jA+1]):
                    iA = self.Ai[pA]
                    if w[iA] < jB+1:
                        Zp[jB+1] = Zp[jB+1] + 1
                        Zi += [iA]
                        x[iA] = B.Ax[pB] * self.Ax[pA]
                        w[iA] = jB+1
                    else:
                        w[iA] = jB+1
                        x[iA] = x[iA] + B.Ax[pB] * self.Ax[pA]
            for pZ in range(Zp[jB], Zp[jB+1]):
                Zi[pZ]
                Zx += [x[Zi[pZ]]]
        return SymMatrix(list(Zp), Zi, Zx, self.m, B.n)



    def __radd__(self, B):
        return self.__add__(B)


    def __add__(self, B):
        B = as_sym_matrix(B)
        Bp = B.Ap
        Bi = B.Ai
        Bx = B.Ax
        Ap = self.Ap
        Ai = self.Ai
        Ax = self.Ax
        m = self.m
        n = self.n

        if (m,n) != (B.m, B.n):
            raise ValueError("Incompatible matrix sizes.")

        w = np.zeros(self.m, dtype=int)
        x = [0] * self.m
        Zp = (self.n+1) * [0]
        Zi = []
        Zx = []
        for jA in range(n):
            Zp[jA+1] = Zp[jA]
            for pA in range(Ap[jA], Ap[jA+1]):
                iA = Ai[pA]
                w[iA] = jA+1 # mark this row as visited
                Zi += [iA] # add this row to the output
                x[iA] = Ax[pA]
                Zp[jA+1] += 1
            for pB in range(Bp[jA], Bp[jA+1]):
                iB = Bi[pB]
                if w[iB] < jA+1: # new row
                    w[iB] = jA+1 # mark this row as visited
                    Zi += [iB] # add this row to the output
                    x[iB] = Bx[pB]
                    Zp[jA+1] += 1
                else:
                    x[iB] += Bx[pB]
            for pZ in range(Zp[jA], Zp[jA+1]):
                iZ = Zi[pZ]
                Zx += [x[iZ]]
                x[iZ] = 0 # x is all zeros
        return SymMatrix(Zp, Zi, Zx, m, n)


    def __rmul__(self, arg):
        arg_scalar = get_scalar(arg)
        self_scalar = get_scalar(self)
        if arg_scalar:
            return self.mul_elem(arg_scalar)
        elif self_scalar:
            return as_sym_matrix(arg).mul_elem(self_scalar)
        else:
            return as_sym_matrix(arg) * self
        


    def __neg__(self):
        return self.mul_elem(SymConst(-1))


    def as_vector(self):
        Ap = [0, self.Ap[-1]]
        Ai = []
        for j in range(self.n):
            for p in range(self.Ap[j], self.Ap[j+1]):
                Ai += [self.Ai[p] + self.m * j]
        return SymMatrix(Ap, Ai, self.Ax, self.m*self.n, 1)


    def as_dense(self):
        m, n = self.size
        Ap = [m * j for j in range(n+1)]
        Ai = [p % m for p in range(m*n)]
        Ax = [SymConst(0.0)] * m * n
        for j in range(n):
            for p in range(self.Ap[j], self.Ap[j+1]):
                i = self.Ai[p]
                p2 = j * m + i
                Ax[p2] = self.Ax[p]
        return SymMatrix(Ap, Ai, Ax, m, n)


def diag(A):
    m, n = A.shape
    if n != 1:
       raise ValueError("argument to diag must be a column vector.")
    Bp = np.zeros((m+1), dtype=int)
    Bi = []
    Bx = []
    for p in range(A.nnz):
        Bi += [A.Ai[p]]
        Bx += [A.Ax[p]]
        Bp[A.Ai[p]+1] = 1
    Bp = list(np.cumsum(Bp))
    return SymMatrix(Bp, Bi, Bx, m, m)


def transpose(A):
    Atp = [0] * (A.m+1)
    Ati = [0] * A.nnz
    Atx = [0] * A.nnz

    for j in range(A.n):
        for p in range(A.Ap[j], A.Ap[j+1]):
            Atp[A.Ai[p]+1] += 1
    for i in range(A.m):
        Atp[i+1] += Atp[i]

    for j in range(A.n):
        for p in range(A.Ap[j], A.Ap[j+1]):
            i = A.Ai[p]
            Ati[Atp[i]] = j
            Atx[Atp[i]] = A.Ax[p]
            Atp[i] += 1
    for i in range(A.m, 0, -1):
        Atp[i] = Atp[i-1]
    Atp[0] = 0

    return SymMatrix(Atp, Ati, Atx, A.n, A.m)


def kron(B, A):
    n = A.n * B.n
    m = A.m * B.m
    nnz_Z = A.nnz * B.nnz

    Zp = [0] * (n+1)
    Zi = [0] * nnz_Z
    Zx = [0] * nnz_Z
    pZ = 0
    # Could be more efficient..
    for jB in range(B.n):
        for pB in range(B.Ap[jB], B.Ap[jB+1]):
            iB = B.Ai[pB]
            for jA in range(A.n):
                jZ = jA + A.n*jB
                for pA in range(A.Ap[jA], A.Ap[jA+1]):
                    Zp[jZ+1] += 1
    for jZ in range(n):
        Zp[jZ+1] += Zp[jZ]


    for jB in range(B.n):
        for pB in range(B.Ap[jB], B.Ap[jB+1]):
            iB = B.Ai[pB]
            for jA in range(A.n):
                jZ = jA + A.n*jB
                for pA in range(A.Ap[jA], A.Ap[jA+1]):
                    iA = A.Ai[pA]
                    iZ = iA + A.m*iB
                    Zi[Zp[jZ]] = iZ
                    Zx[Zp[jZ]] = A.Ax[pA] * B.Ax[pB]
                    Zp[jZ] += 1
    for jZ in range(n, 0, -1):
        Zp[jZ] = Zp[jZ-1]
    Zp[0] = 0

    return SymMatrix(Zp, Zi, Zx, m, n)


def eye(n):
    return NotImplemented


def reciprocals(A):
    Ax = []
    for p in range(A.nnz):
        Ax += [SymConst(1) / A.Ax[p]]
    return SymMatrix(A.Ap, A.Ai, Ax, A.m, A.n)



def get_scalar(expr):
    if isinstance(expr, SymMatrix):
        if expr.shape == (1,1):
            if expr.nnz == 0:
                return SymConst(0)
            else:
                return expr.Ax[0]
        else:
            return None
    elif isinstance(expr, float) or isinstance(expr, int):
        return float(expr)
    elif (isinstance(expr, np.matrix) or
          isinstance(expr, np.ndarray) or
          isinstance(expr, sp.csc_matrix) or
          isinstance(expr, sp.csr_matrix) or
          isinstance(expr, sp.coo_matrix)):
        if expr.shape == (1,1):
            return SymConst(expr[0,0])
        else:
            return None
    else:
        raise TypeError("Unsupported argument type: %s" % str(type(expr)))
                
        
def zero_pad(mat, new_size, insert_idx):
    if (mat.size[0] + insert_idx[0] > new_size[0] or
        mat.size[1] + insert_idx[1] > new_size[1]):
          raise ValueError("Invalid sizes")
    right_pad = new_size[1] - mat.n - insert_idx[1]
    Ap = [0] * insert_idx[1] + mat.Ap + [mat.Ap[-1]] * right_pad
    Ap = [0] * insert_idx[1] + mat.Ap + [mat.Ap[-1]] * right_pad
    Ai = [mat.Ai[p] + insert_idx[0] for p in range(mat.nnz)]
    return SymMatrix(Ap, Ai, mat.Ax, new_size[0], new_size[1])



def block_diag(mats):
    Ap = [0]
    Ai = []
    Ax = []
    m = sum([mat.m for mat in mats])
    n = sum([mat.n for mat in mats])
    j = 0
    for mat in mats:
        for jm in range(mat.n):
            Ap += [Ap[-1]]
            for p in range(mat.Ap[jm], mat.Ap[jm+1]):
                Ap[j+1] += 1
                Ai += [mat.Ai[p]]
                Ax += [mat.Ax[p]]
            j += 1
    return SymMatrix(Ap, Ai, Ax, m, n)


def zeros(m, n):
    return SymMatrix([0 for i in range(n+1)], [], [], m, n)





class SymMatrixIter():
    def __init__(self, A):
        self.A = A
        self.j = 0
        self.p = 0

    def next(self): # For Python 2 compatibility.
        return self.__next__()

    def __next__(self):
        if self.p == self.A.Ap[-1]:
            raise(StopIteration)
        i = self.A.Ai[self.p]
        j = self.j
        p = self.p
        x = self.A.Ax[self.p]
        self.p = self.p + 1
        if self.p == self.A.Ap[self.j+1]:
            self.j = self.j + 1
        return (i,j,p,x)



def as_sym_matrix(expr, sparsity=None):
    if isinstance(expr, SymMatrix):
        return expr
    elif isinstance(expr, CallbackParam):
        return cscparam2sym(expr, sparsity)
    elif isinstance(expr, Parameter):
        return denserowparam2sym(expr)
    elif isinstance(expr, float) or isinstance(expr, int):
        return SymMatrix([0,1], [0], [SymConst(expr)], 1, 1)
    elif isinstance(expr, np.matrix):
        return densemat2sym(expr)
    elif isinstance(expr, np.ndarray):
        return densemat2sym(np.asmatrix(expr))
    elif isinstance(expr, sp.csc_matrix):
        return cscmat2sym(expr)
    elif isinstance(expr, sp.coo_matrix):
        return cscmat2sym(sp.csc_matrix(expr))
    else:
        raise TypeError("Unsupported argument type: %s" % str(type(expr)))



def cscparam2sym(param, sparsity):
    if not isinstance(sparsity, sp.csc_matrix):
        raise TypeError("Second argument must be a Scipy CSC matrix.")
    Ax = []
    for j in range(param.size[1]):
        for p in range(sparsity.indptr[j], sparsity.indptr[j+1]):
            i = sparsity.indices[p]
            Ax += [SymParam(param, (i,j), p)]
    return SymMatrix(list(sparsity.indptr), list(sparsity.indices), Ax,
                     sparsity.shape[0], sparsity.shape[1])


def denserowparam2sym(param):
    m, n = param.size
    Ap = [0]
    Ai = []
    Ax = []
    for j in range(n):
        Ap += [Ap[-1] + param.size[0]]
        for i in range(m):
            Ai += [i]
            nz_idx = i * m + j
            Ax += [SymParam(param, (i,j), nz_idx)]
    return SymMatrix(Ap, Ai, Ax, m, n)


def densemat2sym(mat):
    Ap = [0]
    Ai = []
    Ax = []
    m, n = mat.shape
    for j in range(n):
        Ap += [Ap[-1] + m]
        for i in range(m):
            Ai += [i]
            Ax += [SymConst(mat[i,j])]
    return SymMatrix(Ap, Ai, Ax, m, n)


def cscmat2sym(mat):
    Ax = []
    for j in range(mat.shape[1]):
        for p in range(mat.indptr[j], mat.indptr[j+1]):
            Ax += [SymConst(mat.data[p])]
    return SymMatrix(list(mat.indptr), list(mat.indices), Ax,
                     mat.shape[0], mat.shape[1])
