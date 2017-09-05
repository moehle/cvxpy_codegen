from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy_codegen.param.expr_data import ParamData, ConstData, CbParamData, CONST_ID
from cvxpy_codegen.linop.linop_data import LinOpData, LinOpCoeffData, VarData
from cvxpy.lin_ops.lin_op import SCALAR_CONST, DENSE_CONST, SPARSE_CONST, PARAM, VARIABLE
from cvxpy_codegen.linop.constr_data import ConstrData
import scipy.sparse as sp
from cvxpy_codegen.utils.utils import render



class LinOpHandler():

    
    def __init__(self, sym_data, objective, eq_constr, leq_constr):

        # A list of all variables (not just the named ones).
        self.vars = []
        self.var_ids = []
        
        # List of Coefficients of linear expressions, corresponding to a variable.
        self.linop_coeffs = []

        # The parameters appearing in the linear operators.
        self.linop_params = []
        self.param_ids = []

        # The callback parameters appearing in the linops.
        self.callback_params = []
        self.cbparam_ids = []

        # Constants appearing (as arguments) to the linops.
        self.linop_constants = []

        # A list of the linear operators.
        self.linops = []
        self.linop_ids = []

        # The unique types of linops found in the linop trees.
        self.unique_linops = []

        # The atoms found in the linop trees.
        self.linop_exprs = []
        self.unique_atoms = []

        # Symbolic problem data.
        self.sym_data = sym_data

        # The variables needed to render the C code.
        self.template_vars = dict()

        # Process all the linear expression trees to fill out the
        # lists given above.
        self.objective  = self.process_expression(objective)
        self.eq_constr  = self.process_constr(eq_constr)
        self.leq_constr = self.process_constr(leq_constr)



    # For each constraint, collect data on the constraint,
    # then process its expression trees.
    def process_constr(self, constrs):
        constr_data = []
        vert_offset = 0
        for constr in constrs:
            expr_data = self.process_expression(constr.expr)
            constr_data += [ConstrData(constr, expr_data, vert_offset=vert_offset)]
            vert_offset += expr_data.size[0] * expr_data.size[1] 
        return constr_data


    # Return all variables needed to evaluate the C templates.
    def get_template_vars(self):

        # The sizes of the C workspaces needed to evaluate the
        # the different operations on linops coefficients.
        work_int    = max([c.work_int    for c in self.linop_coeffs])
        work_float  = max([c.work_float  for c in self.linop_coeffs])
        work_coeffs = max([c.work_coeffs for c in self.linop_coeffs])

        # Recover the sparsity patterns of the coefficient matrices.
        obj_coeff, eq_coeff, leq_coeff = self.get_sparsity()

        # Fill out the variables needed to render the C template.
        self.template_vars.update({'vars': self.vars,
                                   'linop_coeffs': self.linop_coeffs,
                                   'linop_constants': self.linop_constants,
                                   'objective': self.objective,
                                   'eq_constr': self.eq_constr,
                                   'leq_constr': self.leq_constr,
                                   'work_int': work_int,
                                   'work_float': work_float,
                                   'work_coeffs': work_coeffs,
                                   'obj_coeff': obj_coeff,
                                   'eq_coeff': eq_coeff,
                                   'leq_coeff': leq_coeff,
                                   'unique_linops': self.unique_linops,
                                   'linop_exprs': self.linop_exprs,
                                   'unique_atoms': self.unique_atoms,
                                   'var_offsets' : self.sym_data.var_offsets,
                                   'CONST_ID' : CONST_ID,
                                   'x_length' : self.sym_data.x_length })

        return self.template_vars



    # Gets the sparsity patterns of the objective and constraint coefficients.
    # (This tells us how much memory to allocate in C).
    def get_sparsity(self):

        # Get Boolean sparse matrix for the objective.
        obj_coeff = self.objective.get_matrix(self.sym_data)
        
        # Get Boolean sparse matrix for the equality constraints.
        eq_coeff = sp.csc_matrix((0,self.sym_data.x_length), dtype=bool)
        for c in self.eq_constr:
            eq_coeff = sp.vstack([eq_coeff, c.get_matrix(self.sym_data)])

        # Get Boolean sparse matrix for the inequality constraints.
        leq_coeff = sp.csc_matrix((0,self.sym_data.x_length), dtype=bool)
        for c in self.leq_constr:
            leq_coeff = sp.vstack([leq_coeff, c.get_matrix(self.sym_data)])

        return (sp.csc_matrix(obj_coeff),
                sp.csc_matrix(eq_coeff),
                sp.csc_matrix(leq_coeff))



    # Recursively process a linear operators,
    # collecting data according to the operator type.
    def process_expression(self, expr):
        if expr.type == PARAM:
            if isinstance(expr.data, CallbackParam):
                if expr.id not in self.cbparam_ids: # Check if already there.
                    data = CbParamData(expr, [data_arg]) # TODO need sparsity pattern of the param.
                    self.callback_params += [data]
                    self.cbparam_ids += [expr.id]

            elif isinstance(expr.data, Parameter):
                if expr.id not in self.param_ids: # Check if already there.
                    data = ParamData(expr)
                    self.linop_params += [data]
                    self.param_ids += [expr.id]

        elif expr.type in [SCALAR_CONST, DENSE_CONST, SPARSE_CONST]:
            data = ConstData(expr)
            self.linop_constants += [data]

        elif expr.type == VARIABLE:
            data = VarData(expr)
            if expr.data not in self.var_ids: # Check if already there.
                self.vars += [data]
                self.var_ids += [expr.data]

        else:  # expr is a linear operator expression:
            if id(expr) in self.linop_ids: # Check if already there.
                idx = self.linop_ids.index(id(expr))
                self.expressions[idx].force_copy()
                data = self.expressions[idx]
            else:
                arg_data = []
                for arg in expr.args: # Recurse on args.
                    arg_data += [self.process_expression(arg)]
                data = LinOpData(expr, arg_data)
                self.linop_coeffs += data.coeffs.values() # Coefficients of variables.
                if data.has_offset:
                    self.linop_exprs += [data.offset_expr] # Expressions of constants.
                self.linops += [data]
                for coeff in data.coeffs.values():
                    if coeff.macro_name not in self.unique_linops:
                         self.unique_linops += [coeff.macro_name]

        return data



    def render(self, target_dir):
        render(target_dir, self.template_vars, 'linop/linop.c.jinja', 'linop.c')



# TODO should this be random? How to make the code deterministic
# TODO Could also move this to utilities, to share with param_handler
def get_val_or_rand(param):
    if not param.value is None:
        return param.value
    else:
        return numpy.ones(param.size)
