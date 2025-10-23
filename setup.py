from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11
import os, subprocess

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Release'
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
        ]
        build_args = ['--config', cfg, '--', '-j4']
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', '..'] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

ext_modules = [Extension('aegaeon_native', sources=[])]

setup(
    name='aegaeon-native',
    version='0.1.0',
    author='OpenAI Research Port',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
