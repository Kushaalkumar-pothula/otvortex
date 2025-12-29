from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'engine',
        ['engine.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3'],
    ),
]

setup(
    name='engine',
    version='1.0',
    author='Your Name',
    description='MHD simulation core in C++',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'],
    zip_safe=False,
)