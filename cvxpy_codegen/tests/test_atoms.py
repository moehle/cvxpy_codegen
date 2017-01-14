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

import unittest
import cvxpy_codegen.tests.utils as tu
import cvxpy_codegen as cg
import numpy as np
import inspect

MODULE = 'cvxpy_codegen.tests.test_atoms'


class TestAtoms(tu.CodegenTestCase):
    class_name = 'TestAtoms'


    def param_setup(self, atom=None):
        np.random.seed(0)
        m = 5
        n = 10
        p = 7
        A = cg.Parameter(m,n, name='A', value=np.random.randn(m,n))
        B = cg.Parameter(n,p, name='B', value=np.random.randn(n,p))
        C = cg.Parameter(m,p, name='C', value=np.random.randn(m,p))
        D = cg.Parameter(p,n, name='D', value=np.random.randn(p,n))
        P = cg.Parameter(n,n, name='P', value=np.random.randn(n,n))
         
        r = cg.Parameter(1,n, name='r', value=np.random.randn(1,n))
        c = cg.Parameter(n,1, name='c', value=np.random.randn(n,1))

        params = [A, B, C, D, P, r, c]

        self.params = dict([(p.name(), p.value) for p in params])


        if   atom == 'abs':
            rhs = cg.abs(A)
        elif atom == 'add':
            rhs = A+A
        elif atom == 'diag_vec':
            rhs = cg.diag(c)
        elif atom == 'diag_mat':
            rhs = cg.diag(P)
        elif atom == 'hstack':
            rhs = cg.hstack(A, C)
        elif atom == 'index':
            rhs = A[0:4:2, 1:9:3]
        elif atom == 'max_entries':
            rhs = cg.max_entries(A)
        elif atom == 'mul':
            rhs = A*B
        elif atom == 'neg':
            rhs = -A
        elif atom == 'reshape':
            rhs = cg.reshape(A, n, m)
        elif atom == 'trace':
            rhs = cg.trace(P)
        elif atom == 'vstack':
            rhs = cg.vstack(A, D)

        x = cg.Variable(rhs.size[0], rhs.size[1], name='x')
        constr = [rhs == x]
        obj = 0

        self.prob = cg.Problem(cg.Minimize(obj), constr)
        self.prob.solve()
        self.x_opt = x.value


    def _test_atom(self, atom_name):
        self.param_setup(atom=atom_name)
        method_name = '_test_' + atom_name
        self._run_codegen_test(self.prob, MODULE, self.class_name, method_name)

    def _run_test_atom(self, atom_name):
        self.param_setup(atom=atom_name)
        from cvxpy_codegen_solver import cg_solve

        param_names = list(inspect.signature(cg_solve).parameters.keys())
        params = dict()
        for p in param_names:
            params[p] = self.params[p]
        var_dict, stats_dict = cg_solve(**params)
        print(self.x_opt)
        print(var_dict['x'])
        self.assertAlmostEqualMatrices(self.x_opt, var_dict['x'])
        

    test_abs          = lambda self: self._test_atom('abs')
    test_add          = lambda self: self._test_atom('add')
    test_diag_vec     = lambda self: self._test_atom('diag_vec')
    test_diag_mat     = lambda self: self._test_atom('diag_mat')
    test_hstack       = lambda self: self._test_atom('hstack')
    test_index        = lambda self: self._test_atom('index')
    test_max_entries  = lambda self: self._test_atom('max_entries')
    test_mul          = lambda self: self._test_atom('mul')
    test_neg          = lambda self: self._test_atom('neg')
    test_reshape      = lambda self: self._test_atom('reshape')
    test_trace        = lambda self: self._test_atom('trace')
    test_vstack       = lambda self: self._test_atom('vstack')


    _test_abs          = lambda self: self._run_test_atom('abs')
    _test_add          = lambda self: self._run_test_atom('add')
    _test_diag_vec     = lambda self: self._run_test_atom('diag_vec')
    _test_diag_mat     = lambda self: self._run_test_atom('diag_mat')
    _test_hstack       = lambda self: self._run_test_atom('hstack')
    _test_index        = lambda self: self._run_test_atom('index')
    _test_max_entries  = lambda self: self._run_test_atom('max_entries')
    _test_mul          = lambda self: self._run_test_atom('mul')
    _test_neg          = lambda self: self._run_test_atom('neg')
    _test_reshape      = lambda self: self._run_test_atom('reshape')
    _test_trace        = lambda self: self._run_test_atom('trace')
    _test_vstack       = lambda self: self._run_test_atom('vstack')




if __name__ == '__main__':
    unittest.main()
