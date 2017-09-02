from cvxpy_codegen.linop.linops.sum import *
from cvxpy_codegen.linop.linops.neg import *
from cvxpy_codegen.linop.linops.index import *


#GET_LINOPDATA = {'sum'  : getdata_sum_lo,
#                 'neg'  : getdata_neg_lo}

GET_LINOPDATA = {'sum'      : getdata_sum_lo,
                 'index'    : getdata_index_lo,
                 'neg'      : getdata_neg_lo}
