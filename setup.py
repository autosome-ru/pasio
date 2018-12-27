from distutils.core import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

gcc_opts = ['-march=native', '-Ofast',]
extensions = [
    Extension(
        "*",
        ["pasio/*.pyx"],
        extra_compile_args=gcc_opts,
        extra_link_args=gcc_opts,
    ),
]

setup(
    ext_modules = cythonize(extensions)
)
