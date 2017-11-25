/*
 *  Copyright 2017 Nicholas Moehle
 *  
 *  This file is part of CVXPY-CODEGEN.
 *  
 *  CVXPY-CODEGEN is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  CVXPY-CODEGEN is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with CVXPY-CODEGEN.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "codegen.h"




void absatom(CsrMatrix *A);

void add(long n_args, CsrMatrix *C, CsrMatrix *Z, long *w, double *x);
void add_coeffs(long n_args, LinopCoeff *C, LinopCoeff *Z, long *w, double *x);

void constant(CsrMatrix *A, long m, long n, long nnz, long *Ap, long *Aj, double *Ax);

void diag_mat(long m, long *Ap, long *Aj, double *Ax, long *Zp, long *Zj, double *Zx);
void diag_vec(long m, long *Ap, long *Aj, double *Ax, long *Zp, long *Zj, double *Zx);

void div(CsrMatrix A, double divisor);
void div_coeffs(LinopCoeff A, double divisor);

void hstack(long n_args, CsrMatrix *C, CsrMatrix *Z, long *offsets);
void hstack_coeffs(long n_args, LinopCoeff *C, LinopCoeff *Z, long *offsets);

void indexatom(long m,
           long start0, long stop0, long step0,
           long start1, long stop1, long step1,
           long *Ap, long *Aj, double *Ax,
           long *Zp, long *Zj, double *Zx);
void lo_index(long m,
           long start0, long stop0, long step0,
           long start1, long stop1, long step1,
           long *Ap, long *Aj, double *Ax,
           long *Zp, long *Zj, double *Zx);



//void kron(long m, long n, long *w, double *x,
//          long *Ap, long *Aj, double *Ax,
//          long *Bp, long *Bj, double *Bx,
//          long *Zp, long *Zj, double *Zx);

void max(long nnz, double *Ax, long *Zp, long *Zj, double *Zx);

void mul(CsrMatrix *A, CsrMatrix *B, CsrMatrix *Z, long *w, double *x);
void mul_coeffs(CsrMatrix *A, LinopCoeff *B, LinopCoeff *Z, long *w, double *x);
// void rmul_coeffs(CsrMatrix *A, LinopCoeff *B, LinopCoeff *Z, long *w, double *x);

void multiply(CsrMatrix *A, CsrMatrix *B, CsrMatrix *Z, long *w, double *x);
void multiply_coeffs(CsrMatrix *A, LinopCoeff *B, LinopCoeff *Z);

void neg(long nnz, double *Ax);
void neg_coeffs(long nnz, double *Ax);

void pos(long nnz, double *Ax);

void promote(CsrMatrix *A, CsrMatrix *Z);
void promote_coeffs(LinopCoeff *A, LinopCoeff *Z);

void reshape(long m, long m_new,
             long *Ap, long *Aj, double *Ax,
             long *Zp, long *Zj, double *Zx);
void reshape_coeffs(LinopCoeff *A, long m, long n);

void sum(CsrMatrix *A, CsrMatrix *Z);
void sum_coeffs(LinopCoeff *A, LinopCoeff *Z, long *w, double *x);

void trace(long m,
         long *Ap, long *Aj, double *Ax,
         long *Zp, long *Zj, double *Zx);


void transpose(long m, long n,
               long *Ap, long *Ai, double *Ax,
               long *Atp, long *Ati, double *Atx);
void transpose_coeffs(LinopCoeff *A, LinopCoeff *Z);

void init_var_coeff(long m_var, long n_var, LinopCoeff *A);
void init_dense_sparsity(CsrMatrix A);

void vstack_coeffs(long n_args, LinopCoeff *C, LinopCoeff *Z, long *offsets);
void vstack(long n_args, CsrMatrix *C, CsrMatrix *Z, long *offsets);
