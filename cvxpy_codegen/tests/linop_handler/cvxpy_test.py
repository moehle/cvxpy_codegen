import cvxpy
import cvxpy_codegen as cg
from cvxpy.problems.solvers.ecos_intf import ECOS
ECOS = ECOS()
import cvxpy.settings as s
from cvxpy_codegen.linop_sym.linop_handler_sym import LinOpHandlerSym
from cvxpy.problems.problem_data.sym_data import SymData

m = 10
n = 20
var_mn = cg.Variable(m, n, name='var_mn')
expr = cg.sum_entries(var_mn)
prob = cvxpy.Problem(cvxpy.Minimize(expr))
obj, constrs = expr.canonicalize()
#cached_data = {}  # Needed for CVXPY.
data = ECOS.get_problem_data(obj, constrs, prob._cached_data)
true_obj_coeff  = data[s.C]
true_obj_offset = data[s.OFFSET]
true_eq_coeff   = data[s.A]
true_eq_offset  = data[s.B]
true_leq_coeff  = data[s.G]
true_leq_offset = data[s.H]

sym_data = SymData(obj, constrs, ECOS)
constr_map = sym_data.constr_map

eq_constrs, leq_constrs, __ = ECOS.split_constr(constr_map)

linop_handler = LinOpHandlerSym(sym_data, obj, eq_constrs, leq_constrs)
template_vars = linop_handler.get_template_vars()
