# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

import subprocess


class CMakeBuild(build_ext):
    def run(self):
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        base_dir = Path(self.get_ext_fullpath(ext.name)).parent.parent
        # Create build files.
        subprocess.check_call(["cmake", base_dir], cwd=self.build_temp)
        # Run the build.
        subprocess.check_call(["cmake", "--build", "."], cwd=self.build_temp)


setup(
    ext_modules=[Extension(name="trt_dft_plugins", sources=[])],
    cmdclass=dict(build_ext=CMakeBuild),
)
