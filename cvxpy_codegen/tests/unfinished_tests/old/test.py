import unittest
import cvxpy_codegen.tests.test_utils as u
import cvxpy_codegen as cg
import numpy
target_dir = "/home/moehle/research/embedded_admm/cvxpy_codegen/c"


obj = cg.Minimize(0)
a = cg.Parameter(1, name='a')
constr = [a == 1]
prob = cg.Problem(obj, constr)
#solver_fun = u.get_solve_fcn(prob, target_dir)
var_dict = u.get_solve_fcn(prob, target_dir)(a=1.0)
