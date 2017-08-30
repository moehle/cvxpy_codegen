# CVXPY-CODEGEN

**WARNING:** This tool is still in an early stage of development, and many bugs might still exist.  Consider it an early alpha, and don't use it for safety-critical applications (yet).

CVXPY-CODEGEN generates embedded C code for solving convex optimization problems.  It allows the user to specify a family of convex optimization problems at a high abstraction level using CVXPY in Python, and then solve instances of this problem family in C (possibly on an embedded microcontroller).  The generated C code is essentially a wrapper for embedded optimization solvers (currently only ECOS) for the specified family of problems.

Abstractly, CVXPY-CODEGEN addresses parametrized *families* of convex optimization problems of the form:

    minimize    f_0(x, a)
    subject to  f_i(x, a) <= 0, for i = 1,...,m.

The parameter `a` defines a specific problem instance in the family; for every such problem instance, the variable `x` is to be determined by solving the optimization problem.  In CVXPY-CODEGEN, the problem family (*ie*, the convex functions `f_i`) are specified in Python using CVXPY.  After C code is generated for this family, the user passes in the parameter `a`, and the problem is solved (all in C).  Currently, problems handled include least squares problems, linear programs (LPs), quadratic programs (QPs), second-order cone programs (SOCPs).

#### Least squares example
To make this all concrete, let's try a simple least-squares problem:

    import cvxpy as cvx
    from cvxpy_codegen import codegen
    m = 10
    n = 5
    A = cvx.Parameter(m, n, name='A')
    b = cvx.Parameter(m, name='b')
    x = cvx.Variable(n, name='x')
    f0 = cvx.norm(A*x - b)
    prob = cvx.Problem(cvx.Minimize(f0))
    codegen(prob, 'least_squares_example')

Then the generated code is available in the `least_squares_example` directory (which is in the currenty directory).  The API is contained in the header file `codegen.h`.  To test out the embedded solver on randomly generated data, run

    cd ~/least_squares_example
    make
    ./example_problem

If you'd rather not use random data, you can specify the data to be used by adding 

    import numpy as np
    A.value = np.random.randn(m, n)
    b.value = np.random.randn(m, 1)

before generating the C code in Python. (Presumably you would replace the random matrices with whatever values you'd like.)

The directory also contains a Python wrapper, so you can use your embedded C solver in Python as a C extension.  To install this C extension, navigate over to the directory with the generated code, and type `python setup.py install`.  To use it, import it with `import cvxpy_codegen_solver`


#### Optimal control example
As a more sophistocated example, we consider a constrained, linear optimal control problem (such as for model predictive control, or MPC).

    import cvxpy as cvx
    from cvxpy_codegen import codegen
    np.random.seed(0)
    n = 5
    m = 3
    T = 15

    A  = cvx.Parameter(n, n, name='A')
    B  = cvx.Parameter(n, m, name='B')
    x0 = cvx.Parameter(n, 1, name='x0')

    x = cvx.Variable(n, T+1, name='x')
    u = cvx.Variable(m, T, name='u')

    obj = 0
    constr = []
    constr += [x[:,0] == x0]
    for t in range(T):
        constr += [x[:,t+1] == A*x[:,t] + B*u[:,t]]
        constr += [cvx.norm(u[:,t], 'inf') <= 1] 
        obj += cvx.sum_squares(x[:,t+1]) + cvx.sum_squares(u[:,t])

    prob = cvx.Problem(cg.Minimize(obj), constr)
    codegen(prob, 'opt_ctrl_example')

#### Installation
To install, clone this repository, `cd` over the directory of the cloned repo, and run `python setup.py install`.  Currently, CVXPY-CODEGEN is not available through any Python repository.  CVXPY-CODEGEN was only tested in Linux with Python 3.5.

#### Limitations
It is *not* possible (and will never be possible) to change the dimensions of the parameters within a single family of convex problems.

Sparse parameters are not currently supported.

Due to the way CVXPY currently works, it's not possible to use a parameter as the positive semidefinite matrix in the `quad_form` atom. (As a partial fix, we *can* use `sum_squares(L*x)`, using the Cholesky factor `L` as a parameter instead of the positive semidefinite matrix itself.)

#### License
CVXPY-CODEGEN is currently licensed under GPL version 3.  This is because the only supported backend solver, ECOS, is under GPL version 3.  (If you have a different license for ECOS, I'd be more than happy to provide a more permissive license for CVXPY-CODEGEN.)  I am planning on adding at least one more solver, in which case the license for the generated code would have the most permissive license compatible with the chosen backend solver.
