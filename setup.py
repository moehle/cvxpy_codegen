from setuptools import setup
from glob import glob

ecos_sources = ['ecos/src/*.c',
                'ecos/external/ldl/src/ldl.c',
                'ecos/external/amd/src/*.c',
                'ecos/include/*.h',
                'ecos/external/amd/include/*.h',
                'ecos/external/ldl/include/*.h',
                'ecos/external/ldl/Makefile',
                'ecos/external/SuiteSparse_config/*.h',
                'ecos/external/amd/include/*.h',
                'ecos/external/amd/Makefile',
                'ecos/Makefile',
                'ecos/ecos.mk' ]

atom_dirs = [d.replace('/', '.')[:-1] for d in glob('cvxpy_codegen/atoms/*/')]
package_data={'cvxpy_codegen.atoms'                : ['atoms.c.jinja', 'linops.c.jinja'],
              'cvxpy_codegen.expr_handler.tree'    : ['*.c.jinja'],
              'cvxpy_codegen.expr_handler.explicit': ['*.c.jinja'],
              'cvxpy_codegen.tests'                : ['*.c.jinja'],
              'cvxpy_codegen.solvers'              : ['*.jinja'] + ecos_sources,
              'cvxpy_codegen.utils'                : ['utils.c.jinja'],
              'cvxpy_codegen.templates'            : ['*.jinja', '*.h', '*.c']}
package_data.update({d : ['*.jinja'] for d in atom_dirs})


setup(
    name='cvxpy-codegen',
    version='0.0.1',
    author='Nicholas Moehle',
    author_email='moehle@stanford.edu',
    packages=['cvxpy_codegen',
              'cvxpy_codegen.atoms',
              'cvxpy_codegen.expr_handler',
              'cvxpy_codegen.expr_handler.explicit',
              'cvxpy_codegen.expr_handler.tree',
              'cvxpy_codegen.object_data',
              'cvxpy_codegen.solvers',
              'cvxpy_codegen.templates',
              'cvxpy_codegen.tests',
              'cvxpy_codegen.tests.examples',
              'cvxpy_codegen.utils']
              + atom_dirs,
    package_dir={'cvxpy_codegen': 'cvxpy_codegen'},
    package_data=package_data,
    url='http://github.com/moehle/cvxpy-codegen/',
    license='GPLv3',
    zip_safe=False,
    description='Embedded C code generation for convex optimization' + 
                'using CVXPY.',
    install_requires=["ecos >= 2",
                      "cvxpy >= 1.0",
                      "numpy >= 1.9",
                      "jinja2 >= 2.8",
                      "scipy >= 0.15"],
)
