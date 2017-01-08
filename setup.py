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

setup(
    name='cvxpy-codegen',
    version='0.0.1',
    author='Nicholas Moehle',
    author_email='moehle@stanford.edu',
    packages=['cvxpy_codegen',
              'cvxpy_codegen.atoms',
              'cvxpy_codegen.linop_sym',
              'cvxpy_codegen.param',
              'cvxpy_codegen.solvers',
              'cvxpy_codegen.templates',
              'cvxpy_codegen.tests',
              'cvxpy_codegen.utils'], # TODO remove utils dir
    package_dir={'cvxpy_codegen': 'cvxpy_codegen'},
    package_data={'cvxpy_codegen.param'             : ['param.c.jinja'],
                  'cvxpy_codegen.atoms'             : ['*.jinja'],
                  'cvxpy_codegen.linop_sym'         : ['linop_sym.c.jinja'],
                  'cvxpy_codegen.solvers'           : ['*.jinja'] + ecos_sources,
                  'cvxpy_codegen.utils'             : ['utils.c.jinja'],
                  'cvxpy_codegen.templates'  : ['*.jinja']},
    url='http://github.com/cvxgrp/cvxpy-codegen/', # TODO
    license='GPLv3',
    zip_safe=False, # TODO
    description='Embedded C code generation for convex optimization problem using CVXPY.',
    install_requires=["ecos >= 2",
                      "cvxpy >= 0.4", # TODO check
                      "numpy >= 1.9",
                      "scipy >= 0.15"],
)
