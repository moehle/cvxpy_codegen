import unittest
import cvxpy_codegen.tests.test_utils as tu
import cvxpy_codegen as cg
import numpy

MODULE = 'cvxpy_codegen.tests.problem_test'


class FeasProbTest(tu.CodegenTestCase):
    class_name = 'FeasProbTest'


    def test_feasprob(self):
        y = cg.Variable(1, name='y')
        obj = cg.Minimize(0)
        constr = [y == 2]
        prob = cg.Problem(obj, constr)
        self.run_codegen_test(prob, MODULE, self.class_name, '_test_feasprob')

    @classmethod
    def _test_feasprob(cls):
        from cvxpy_codegen_solver import cg_solve
        var_dict = cg_solve()
        assert abs(var_dict['y'][0] - 2.0) <= 1e-5
    

    def test_noparams(self):
        x = cg.Variable(1, name='x')
        obj = cg.Minimize(x)
        constr = [x >= 3]
        prob = cg.Problem(obj, constr)
        self.run_codegen_test(prob, MODULE, self.class_name, '_test_noparams')

    @classmethod
    def _test_noparams(cls):
        from cvxpy_codegen_solver import cg_solve
        var_dict = cg_solve()
        assert abs(var_dict['x'][0] - 3.0) <= 1e-5
    



if __name__ == '__main__':
    unittest.main()
