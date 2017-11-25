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





void absatom(CsrMatrix *A){
  long p, nnz=A->nnz;
  double x;
  for(p=0; p<nnz; p++){
    x = A->x[p];
    A->x[p] = x>0 ? x : -x;
  }
}





void add(long n_args, CsrMatrix *C, CsrMatrix *Z, long *w, double *x){
  long i, j, k, p, p2, q=0, m=0, n=0;
  for(k=0; k<n_args; k++) m = m>C[k].m ? m : C[k].m;
  for(k=0; k<n_args; k++) n = n>C[k].n ? n : C[k].n;
  float alpha;
  Z->p[0] = 0;
  for(i=0; i<n; i++) w[i] = 0;
  for(i=0; i<m; i++){
    Z->p[i+1] = Z->p[i];
    for(k=0; k<n_args; k++){
      if(C[k].m==1 && C[k].n==1){ /* This is scalar-matrix addition. */
        alpha = C[k].x[0];
        for(j=0; j<n; j++){
          if(w[j] < i+1){
            w[j] = i+1;
            Z->j[q++] = j;
            x[j] = alpha;
            Z->p[i+1]++;
          }
          else{
            x[j] += alpha;
          }
        }
      }
      else {  /* This is matrix-matrix addition. */
        p2 = C[k].p[i+1];
        for(p=C[k].p[i]; p<p2; p++){
          j = C[k].j[p];
          if(w[j] < i+1){
            w[j] = i+1;
            Z->j[q++] = j;
            x[j] = C[k].x[p];
            Z->p[i+1]++;
          }
          else{
            x[j] += C[k].x[p];
          }
        }
      }
      p2 = Z->p[i+1];
      for(p=Z->p[i]; p<p2; p++) Z->x[p] = x[Z->j[p]];
    }
  }
  Z->m = m;
  Z->n = n;
  Z->nnz = Z->p[m];
}




void add_coeffs(long n_args, LinopCoeff *C, LinopCoeff *Z, long *w, double *x){
  long i, j, k, p, p2, q=0, m=C[0].m, n=C[0].n;
  Z->p[0] = 0;
  for(i=0; i<n; i++) w[i] = 0;
  for(i=0; i<m; i++){
    Z->p[i+1] = Z->p[i];
    for(k=0; k<n_args; k++){
      p2 = C[k].p[i+1];
      for(p=C[k].p[i]; p<p2; p++){
        j = C[k].j[p];
        if(w[j] < i+1){
          w[j] = i+1;
          Z->j[q++] = j;
          x[j] = C[k].x[p];
          Z->p[i+1]++;
        }
        else{
          x[j] += C[k].x[p];
        }
      }
    }
    p2 = Z->p[i+1];
    for(p=Z->p[i]; p<p2; p++) Z->x[p] = x[Z->j[p]];
  }
  Z->m = m;
  Z->n = n;
  Z->m_var = C[0].m_var;
  Z->n_var = C[0].n_var;
}







void diag_mat(long m,
              long *Ap, long *Aj, double *Ax,
              long *Zp, long *Zj, double *Zx){
  long i, j, p, p2, count=0;

  for (i=0; i<=m; i++)  Zp[i] = 0;

  for (i=0; i<m; i++){
    p2 = Ap[i+1];
    for (p=Ap[i]; p<p2; p++){
      j = Aj[p];
      if (i==j){
        Zp[i+1]++;
        Zj[count] = 0l;
        Zx[count++] = Ax[p];
      }
    }
  }

  for (i=0; i<m; i++)  Zp[i+1] += Zp[i];
}





void diag_vec(long m,
          long *Ap, long *Aj, double *Ax,
          long *Zp, long *Zj, double *Zx){
  long i, p, p2;

  for (i=0; i<=m; i++)  Zp[i] = Ap[i];
  for (i=0; i<m; i++){
    p2 = Ap[i+1];
    for (p=Ap[i]; p<p2; p++){
      Zj[p] = i;
      Zx[p] = Ax[p];
    }
  }
}




void div(CsrMatrix A, double divisor){
  long i, nnz=A.nnz;
  for(i=0; i<nnz; i++){
    A.x[i] = A.x[i] / divisor;
  }
}



void div_coeffs(LinopCoeff A, double divisor){
  long i, nnz=A.nnz;
  for(i=0; i<nnz; i++){
    A.x[i] = A.x[i] / divisor;
  }
}



void hstack(long n_args, CsrMatrix *C, CsrMatrix *Z, long *offsets){
  long i, k, p, p2, count=0;

  Z->p[0] = 0;
  for (i=0; i<Z->m; i++){
    Z->p[i+1] = 0;
    for (k=0; k<n_args; k++){
      p2 = C[k].p[i+1];
      Z->p[i+1] += p2;
      for (p=C[k].p[i]; p<p2; p++){
        Z->j[count]   = C[k].j[p] + offsets[k];
        Z->x[count++] = C[k].x[p];
      }
    }
  }
}



void hstack_coeffs(long n_args, LinopCoeff *C, LinopCoeff *Z, long *offsets){
  long i, k, p, p2, count=0, iZ=0;

  Z->p[iZ++] = 0;
  for (k=0; k<n_args; k++){
    for ( ; iZ<offsets[k]+1; iZ++) Z->p[iZ] = Z->p[iZ-1];
    for (i=0; i<C[k].m; i++){
      p2 = C[k].p[i+1];
      Z->p[iZ] = Z->p[iZ-1] + p2 - C[k].p[i];
      for (p=C[k].p[i]; p<p2; p++){
        Z->j[count]   = C[k].j[p];
        Z->x[count++] = C[k].x[p];
      }
      iZ++;
    }
  }
  for ( ; iZ<=Z->m; iZ++) Z->p[iZ] = Z->p[iZ-1];
}



void indexatom(long m,
           long start0, long stop0, long step0,
           long start1, long stop1, long step1,
           long *Ap, long *Aj, double *Ax,
           long *Zp, long *Zj, double *Zx){
  long i, j, i_new=0, i_new_pp, j_new, p, p2;

  Zp[0] = 0l;
  for (i=start0; i<stop0; i+=step0){
    p2 = Ap[i+1];
    i_new_pp = i_new + 1l;
    Zp[i_new_pp] = Zp[i_new];
    for (p=Ap[i]; p<p2; p++){
      j = Aj[p];
      if (j>=start1 && j<stop1 && (j-start1) % step1 == 0){
        j_new = (j-start1) / step1;
        Zj[Zp[i_new_pp]] = j_new;
        Zx[Zp[i_new_pp]] = Ax[p];
        Zp[i_new_pp]++;
      }
    }
    i_new++;
  }
}



void lo_index(long m,
           long start0, long stop0, long step0,
           long start1, long stop1, long step1,
           long *Ap, long *Aj, double *Ax,
           long *Zp, long *Zj, double *Zx){

  int i, j, idx, p, p2, pZ=0, iZ=0;
  Zp[0] = 0;
  for (j=start1; j<stop1; j+=step1){
    for (i=start0; i<stop0; i+=step0){
      Zp[iZ+1] = Zp[iZ];
      idx = i + j * m;
      p2 = Ap[idx+1];
      for (p=Ap[idx]; p<p2; p++){
        Zj[pZ] = Aj[p];
        Zx[pZ++] = Ax[p];
        Zp[iZ+1]++;
      }
      iZ++;
    }
  }
}




/* Computes the Kronecker product Z = kron(A, B). */
/* TODO wrong, doesn't complile:
void kron(long m, long n, long *w, double *x,
          long *Ap, long *Aj, double *Ax,
          long *Bp, long *Bj, double *Bx,
          long *Zp, long *Zj, double *Zx){

  long i, j, k, p, p2, q, q2, r=0;
  double Aij;
  for(i=0; i<n; i++)  w[i] = 0;
  Zp[0] = 0;

  for(i=0; i<nZ; i++){
    Zp[i+1] = Zp[i];
    p2 = Ap[i+1];
    for(p=Ap[i]; p<p2; p++){
      j = Aj[p];
      Aij = Ax[p];
      q2 = Bp[j+1];
      for(q=Bp[j]; q<q2; q++){
        k = Bj[q];
        if(w[k] < i+1){
          w[k] = i+1;
          x[k] = Aij * Bx[q];
          Zj[r++] = k;
          Zp[i+1]++;
        }
        else{
          x[k] += Aij * Bx[q];
        }
      }
    }
    for(p=Zp[i]; p<Zp[i+1]; p++){
      Zx[p] = x[Zj[p]];
    }
  }
}
*/




void max(long nnz, double *Ax, long *Zp, long *Zj, double *Zx){
  long i;
  double x, max = Ax[0];
  for(i=0; i<nnz; i++){
    x = Ax[i];
    if(x > max){
      max = x;
    }
  }
  Zp[0] = 0;
  Zp[1] = 1;
  Zj[0] = 0;
  Zx[0] = max;
}



void mul(CsrMatrix *A, CsrMatrix *B, CsrMatrix *Z, long *w, double *x){

  long i, j, k, p, p2, q, q2, r=0, m=Z->m, n=Z->n;
  double Aij;
  for(i=0; i<n; i++)  w[i] = 0;
  Z->p[0] = 0;
  for(i=0; i<m; i++){
    Z->p[i+1] = Z->p[i];
    p2 = A->p[i+1];
    for(p=A->p[i]; p<p2; p++){
      j = A->j[p];
      Aij = A->x[p];
      q2 = B->p[j+1];
      for(q=B->p[j]; q<q2; q++){
        k = B->j[q];
        if(w[k] < i+1){
          w[k] = i+1;
          x[k] = Aij * B->x[q];
          Z->j[r++] = k;
          Z->p[i+1]++;
        }
        else{
          x[k] += Aij * B->x[q];
        }
      }
    }
    for(p=Z->p[i]; p<Z->p[i+1]; p++){
      Z->x[p] = x[Z->j[p]];
    }
  }
}



void mul_coeffs(CsrMatrix *A, LinopCoeff *B, LinopCoeff *Z,
                long *w, double *x){

  long i, j, k, iZ, jZ, iB, p, p2, q, q2, r=0, n=Z->n;
  double Aij;
  for(i=0; i<n; i++)  w[i] = 0;
  Z->p[0] = 0;
  for (jZ=0; jZ<Z->n_var; jZ++){
    for(i=0; i<Z->m_var; i++){
      iZ = jZ * Z->m_var + i;
      Z->p[iZ+1] = Z->p[iZ];
      p2 = A->p[i+1];
      for(p=A->p[i]; p<p2; p++){
        j = A->j[p];
        Aij = A->x[p];
        iB = j + jZ * B->m_var;
        q2 = B->p[iB+1];
        for(q=B->p[iB]; q<q2; q++){
          k = B->j[q];
          if(w[k] < i+1){
            w[k] = i+1;
            x[k] = Aij * B->x[q];
            Z->j[r++] = k;
            Z->p[iZ+1]++;
          }
          else{
            x[k] += Aij * B->x[q];
          }
        }
      }
      for(p=Z->p[iZ]; p<Z->p[iZ+1]; p++){
        Z->x[p] = x[Z->j[p]];
      }
    }
  }
}



/* TODO unfinished
void rmul_coeffs(CsrMatrix *A, LinopCoeff *B, LinopCoeff *Z,
                long *w, double *x){

  long i, j, k, iZ, jZ, iB, p, p2, q, q2, r=0, n=Z->n;
  double Aij;
  for(i=0; i<n; i++)  w[i] = 0;
  Z->p[0] = 0;
  for (jZ=0; jZ<Z->n_var; jZ++){
    for(i=0; i<Z->m_var; i++){
      iZ = jZ * Z->m_var + i;
      Z->p[iZ+1] = Z->p[iZ];
      p2 = A->p[i+1];
      for(p=A->p[i]; p<p2; p++){
        j = A->j[p];
        Aij = A->x[p];
        iB_start = j * B->m_var;
        iB_stop = (j+1) * B->m_var;
        for (iB=iB_start; iB<iB_stop; iB++){
          q2 = B->p[iB+1];
          for(q=B->p[iB]; q<q2; q++){
            k = B->j[q];
            if(w[k] < i+1){
              w[k] = i+1;
              x[k] = Aij * B->x[q];
              Z->j[r++] = k;
              Z->p[iZ+1]++;
            }
            else{
              x[k] += Aij * B->x[q];
            }
          }
        }
      }
      for(p=Z->p[iZ]; p<Z->p[iZ+1]; p++){
        Z->x[p] = x[Z->j[p]];
      }
    }
  }
}
*/




void multiply(CsrMatrix *A, CsrMatrix *B, CsrMatrix *Z,
              long *w, double *x){
  long i, j, p, p2, q=0, m=Z->m;

  Z->p[0] = 0;
  for(i=0; i<m; i++){
    Z->p[i+1] = Z->p[i];
    p2 = A->p[i+1];
    for(p=A->p[i]; p<p2; p++){
      j = A->j[p];
      w[j] = i+1;
      x[j] = A->x[p];
    }
    p2 = B->p[i+1];
    for(p=B->p[i]; p<p2; p++){
      j = B->j[p];
      if(w[j] == i+1){
        Z->x[q] = x[j] * B->x[p];
        Z->j[q++] = j;
        Z->p[i+1]++;
      }
    }
  }
}




void multiply_coeffs(CsrMatrix *A, LinopCoeff *B, LinopCoeff *Z){

  long i, j, iB, p, p2, q, m=Z->m, m_var=Z->m_var;
  double x;

  for (i=0; i<=m; i++) Z->p[i] = 0;
  for(i=0; i<m_var; i++){
    p2 = A->p[i+1];
    for(p=A->p[i]; p<p2; p++){
      j = A->j[p];
      iB = j*m_var + i;
      Z->p[iB+1] +=  B->p[iB+1] - B->p[iB];
    }
  }
  for (i=0; i<m; i++) Z->p[i+1] += Z->p[i];
  for(i=0; i<m_var; i++){
    p2 = A->p[i+1];
    for(p=A->p[i]; p<p2; p++){
      j = A->j[p];
      x = A->x[p];
      iB = j*m_var + i;
      for(q=B->p[iB]; q<B->p[iB+1]; q++){
        Z->x[Z->p[iB]] = B->x[q] * x;
        Z->j[Z->p[iB]] = B->j[q];
        Z->p[iB]++;
      }
    }
  }
  for (i=m; i>0; i--) Z->p[i] = Z->p[i-1];
  Z->p[0] = 0;
}




void neg(long nnz, double *Ax){
  long i;
  for(i=0; i<nnz; i++){
    Ax[i] = -Ax[i];
  }
}



void neg_coeffs(long nnz, double *Ax){
  long i;
  for(i=0; i<nnz; i++){
    Ax[i] = -Ax[i];
  }
}




void pos(long nnz, double *Ax){
  long i;
  double x;
  for(i=0; i<nnz; i++){
    x = Ax[i];
    Ax[i] = x ? x>0 : 0.0;
  }
}



void promote(CsrMatrix *A, CsrMatrix *Z){
  long i, j, m=Z->m, n=Z->n, count=0;

  Z->p[0] = 0;
  for(i=0; i<m; i++){
    Z->p[i+1] = Z->p[i];
    for(j=0; j<n; j++){
      Z->j[count] = j;
      Z->x[count++] = A->x[0];
      Z->p[i+1]++;
    }
  }
}



void promote_coeffs(LinopCoeff *A, LinopCoeff *Z){
  long i, p, p2, m=Z->m, count=0;

  Z->p[0] = 0;
  for(i=0; i<m; i++){
    Z->p[i+1] = Z->p[i];
    p2 = A->p[1];
    for(p=0; p<p2; p++){
      Z->j[count] = A->j[p];
      Z->x[count++] = A->x[p];
      Z->p[i+1]++;
    }
  }
}



void reshape(long m, long m_new,
             long *Ap, long *Aj, double *Ax,
             long *Zp, long *Zj, double *Zx){
  long i, j, i_new, j_new, idx, p, p2;

  for (i=0; i<=m_new; i++)  Zp[i] = 0;

  for (i=0; i<m; i++){
    p2 = Ap[i+1];
    for (p=Ap[i]; p<p2; p++){
      j = Aj[p];
      idx = j*m + i;
      i_new = idx % m_new;
      Zp[i_new+1]++;
    }
  }

  for (i=0; i<m_new; i++)  Zp[i+1] += Zp[i];

  for (i=0; i<m; i++){
    p2 = Ap[i+1];
    for (p=Ap[i]; p<p2; p++){
      j = Aj[p];
      idx = j*m + i;
      i_new = idx % m_new;
      idx -= i_new;
      j_new = idx/m_new;
      Zj[Zp[i_new]] = j_new;
      Zx[Zp[i_new]] = Ax[p];
      Zp[i_new]++;
    }
  }

  for (i=m_new; i>0; i--)  Zp[i] = Zp[i-1];
  Zp[0] = 0l;
}



void reshape_coeffs(LinopCoeff *A, long m, long n){
    A->m_var = m;
    A->n_var = n;
}


void sum(CsrMatrix *A, CsrMatrix *Z){
  long i, p, p2, m=A->m;
  Z->p[0] = 0;
  Z->p[1] = 1;
  Z->j[0] = 0;
  Z->x[0] = 0;
  Z->m = 1;
  Z->n = 1;
  Z->nnz = 1;
  for(i=0; i<m; i++){
    p2 = A->p[i+1];
    for(p=A->p[i]; p<p2; p++){
      Z->x[0] += A->x[p];
    }
  }
}




void sum_coeffs(LinopCoeff *A, LinopCoeff *Z, long *w, double *x){
  long i, j, p, p2, q=0, m=A->m, n=A->n;

  Z->p[0] = 0;
  Z->p[1] = 0;
  Z->j[0] = 0;
  Z->m = 1;
  Z->n = n;
  Z->m_var = 1;
  Z->n_var = 1;

  for(j=0; j<n; j++){
    w[j] = 0;
    x[j] = 0;
  }
  for(i=0; i<m; i++){
    p2 = A->p[i+1];
    for(p=A->p[i]; p<p2; p++){
      j = A->j[p];
      if(w[j] == 0){
        w[j] = 1;
        Z->j[q++] = j;
        x[j] = A->x[p];
        Z->p[1]++;
      }
      else{
        x[j] += A->x[p];
      }
    }
  }
  p2 = Z->p[1];
  for(p=0; p<p2; p++) Z->x[p] = x[Z->j[p]];
}




/* Computes Z = A*B. */
void trace(long m,
           long *Ap, long *Aj, double *Ax,
           long *Zp, long *Zj, double *Zx){
  long i, p, p2;
  double trace = 0;

  for(i=0; i<m; i++){
    p2 = Ap[i+1];
    for(p=Ap[i]; p<p2; p++){
      if (i == Aj[p])  trace += Ax[p];
    }
  }
  Zp[0] = 0;
  Zp[1] = 1;
  Zj[0] = 0;
  Zx[0] = trace;
}



void transpose(long m, long n,
               long *Ap, long *Ai, double *Ax,
               long *Atp, long *Ati, double *Atx){
    long i, j, k, ind;


    for (k=0; k<=m; k++)  Atp[k] = 0;
    for (k=0; k<Ap[n]; k++) Atp[Ai[k]+1]++;
    for (j=0; j<m; j++) Atp[j+1] += Atp[j];

    for (j=0; j<n; j++){
        for (k=Ap[j]; k<Ap[j+1]; k++){
            i = Ai[k];
            ind = Atp[i];
            Ati[ind] = j;
            Atx[ind] = Ax[k];
            Atp[i]++;
        }
    }
    for (j=m; j>0; j--) Atp[j] = Atp[j-1];
    Atp[0] = 0;
}




void transpose_coeffs(LinopCoeff *A, LinopCoeff *Z){
  long iA, iA_var, jA_var, mA_var=A->m_var;
  long iZ, iZ_var, jZ_var, mZ_var=Z->m_var, mZ=Z->m;
  long p, p2, count=0;

  Z->p[0] = 0;
  for (iZ=0; iZ<mZ; iZ++){
    iZ_var = iZ % mZ_var;
    jZ_var = iZ / mZ_var;
    iA_var = jZ_var;
    jA_var = iZ_var;
    iA = iA_var + jA_var * mA_var;
    p2 = A->p[iA+1];
    for (p=A->p[iA]; p<p2; p++){
      Z->j[count] = A->j[p];
      Z->x[count++] = A->x[p];
      // Zp[iZ+1]++;
    }
    Z->p[iZ+1] = Z->p[iZ] + p2 - A->p[iA];
  }
}





void vstack(long n_args, CsrMatrix *C, CsrMatrix *Z, long *offsets){
  long i, k, p, p2, count=0, iZ=0;

  Z->p[iZ++] = 0;
  for (k=0; k<n_args; k++){
    for ( ; iZ<offsets[k]+1; iZ++) Z->p[iZ] = Z->p[iZ-1];
    for (i=0; i<C[k].m; i++){
      p2 = C[k].p[i+1];
      Z->p[iZ] = Z->p[iZ-1] + p2 - C[k].p[i];
      for (p=C[k].p[i]; p<p2; p++){
        Z->j[count]   = C[k].j[p];
        Z->x[count++] = C[k].x[p];
      }
      iZ++;
    }
  }
  for ( ; iZ<=Z->m; iZ++) Z->p[iZ] = Z->p[iZ-1];
}




void vstack_coeffs(long n_args, LinopCoeff *C, LinopCoeff *Z, long *offsets){
  long i, jZ, k, p, p2, iZ=0, count=0;

  Z->p[iZ++] = 0;
  for (jZ=0; jZ<Z->n_var; jZ++){
    for (k=0; k<n_args; k++){
      for ( ; iZ<jZ*Z->m_var+offsets[k]+1; iZ++) Z->p[iZ] = Z->p[iZ-1];
      for (i=jZ*C[k].m_var; i<(jZ+1)*C[k].m_var; i++){
        p2 = C[k].p[i+1];
        Z->p[iZ] = Z->p[iZ-1] + p2 - C[k].p[i];
        for (p=C[k].p[i]; p<p2; p++){
          Z->j[count]   = C[k].j[p];
          Z->x[count++] = C[k].x[p];
        }
        iZ++;
      }
    }
  }
  for ( ; iZ<=Z->m; iZ++) Z->p[iZ] = Z->p[iZ-1];
}




void constant(CsrMatrix *A, long m, long n, long nnz,
                  long *Ap, long *Aj, double *Ax){
    A->m = m;
    A->n = n;
    A->nnz = nnz;
    A->p = Ap;
    A->j = Aj;
    A->x = Ax;
}


void init_var_coeff(long m_var, long n_var, LinopCoeff *A){
    long m = m_var*n_var;
    A->m_var = m_var;
    A->n_var = n_var;
    A->m = m;
    A->n = m;
    A->nnz = m;
    long i;
    A->p[0] = 0;
    for(i=0; i<m; i++){
        A->p[i+1] = A->p[i]+1;
        A->j[i] = i;
        A->x[i] = 1.0;
    }
}



void init_dense_sparsity(CsrMatrix A){
    long i, j, m=A.m, n=A.n;
    for(i=0; i<m; i++){
        A.p[i] = i*n;
        for(j=0; j<n; j++){
            A.j[i * n + j] = j;
        }
        A.p[m] = m*n;
    }
}
