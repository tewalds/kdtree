from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "kdtree",
        ["kdtree_bindings.cpp"],
        include_dirs=[pybind11.get_include()],
        cxx_std=20,
        extra_compile_args=["-O3", "-Wall", "-Wextra"],
    ),
]

setup(
    name="kdtree",
    version="1.0.0",
    author="Your Name",
    description="Dynamic 2D spatial index with insert/remove support",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=["pybind11>=2.10.0"],
    license_files = ('LICENSE',),
)
