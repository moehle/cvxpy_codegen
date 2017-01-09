import cvxpy_codegen as cg
import numpy
import cvxpy_codegen.tests.test_utils as tu
from cvxpy_codegen.templates.template_handler import PKG_PATH
from cvxpy_codegen.utils.utils import FILE_SEP

numpy.random.seed(0)

target_dir = PKG_PATH + FILE_SEP + "c"

# Define problem.
n = 5
m = 3
x = cg.Variable(n, name='x')
B = cg.Parameter(m,n, name='B')
C = cg.Parameter(n,n, name='C')
b = cg.Parameter(m,1, name='b')
b1 = cg.Parameter(m-2,1, name='b1')
b2 = cg.Parameter(1,1, name='b2')
b3 = cg.Parameter(1,1, name='b3')

B.value = numpy.random.randn(m,n)
C.value = numpy.eye(n)
b1.value = numpy.ones((m-2,1))
b2.value = 2
b3.value = 3

b = cg.vstack(b1,b2,b3)
#b = cg.vstack(b1,b2)
#b = numpy.ones((m,1))


#D = -(B*C)
D = -(B*(C+C+C))
constraints = [-numpy.ones((n,1))*cg.trace(C) <= x, x <= numpy.ones((n,1))*4, D*x==b]
#constraints = [-numpy.ones((n,1))*4 <= x, x <= numpy.ones((n,1))*4, D*x==b]
obj = cg.Minimize(x[0])
prob = cg.Problem(obj, constraints)

prob.solve()

# Build and install custom solver.
prob.codegen(target_dir)
tu.install_custom_solver(target_dir)
from cvxpy_codegen_solver import cg_solve
vars_dict = cg_solve(B=B.value, C=C.value, b1=b1.value, b2=b2.value, b3=b3.value)
#vars_dict = cg_solve(B=B.value, C=C.value, b1=b1.value, b2=b2.value)
#vars_dict = cg_solve(B=B.value, C=C.value)

# Print vars
print("opt value is: ", prob.value)
print("\n\n")
print(vars_dict['x'])
print(x.value)



data = prob.get_problem_data(solver=cg.ECOS)
#print(data.keys())
#print('A = ')
#print(data['A'].todense())
#print('b = ')
#print(data['b'])
#print('G = ')
#print(data['G'].todense())
#print('h = ')
#print(data['h'])
#print('c = ')
#print(data['c'])
#print('offset = ')
#print(data['offset'])

print('A.colptr = ')
print(data['A'].indptr)
print('A.rowidx= ')
print(data['A'].indices)
print('A.nzval = ')
print(data['A'].data)
