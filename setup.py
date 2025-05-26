# Copyright 2020-2025 Qihoo Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys

from setuptools import setup


class DistCfg:
    EXT_NAME = "tensornet.core._pywrap_tn"
    PLAT = "manylinux2010_x86_64"
    BUILD_PRESET = "release"
    BUILD_DIR = "build/release"


if "PIXI_IN_SHELL" in os.environ:
    from setuptools import Extension
    from setuptools.command.bdist_wheel import bdist_wheel
    from setuptools.command.build_ext import build_ext
    from setuptools.command.egg_info import egg_info
    from setuptools.command.install_lib import install_lib
    from setuptools.command.install_scripts import install_scripts
    from twine.commands.check import check

    class CMakeBuild(build_ext):
        def build_extensions(self):
            subprocess.check_call(["cmake", "--workflow", "--preset", DistCfg.BUILD_PRESET, "--fresh"])
            # Make sure CMake generated all the appropriate extensions
            for ext in self.extensions:
                ext = self.get_ext_fullpath(ext.name)
                if not os.path.exists(ext):
                    raise RuntimeError(f"Failed to build extension: {ext}")

    class CMakeInstall(install_lib):
        def install(self):
            subprocess.check_call(
                ["cmake", "--build", "--preset", DistCfg.BUILD_PRESET, "--target", "install", "--", "SKIP_RUN_TESTS=1"]
            )

    class NoopInstallScript(install_scripts):
        def run(self):
            pass

    class EggInfo(egg_info):
        def _get_egg_basename(self, py_version=sys.version_info.major, platform=None):
            if py_version:
                return super()._get_egg_basename("3.5", platform)
            else:
                return super()._get_egg_basename(py_version, platform)

    class BdistWheelWithCheck(bdist_wheel):
        def get_tag(self):
            _, _, plat_tag = super().get_tag()
            return "cp35", "abi3", plat_tag

        def run(self):
            # from cmake 3.29, pass install prefix via CMAKE_INSTALL_PREFIX env
            os.putenv("CMAKE_INSTALL_PREFIX", os.path.abspath(self.bdist_dir))

            # build and create the whl package
            super().run()

            # twine check
            impl_tag, abi_tag, plat_tag = self.get_tag()
            whl_file = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}.whl"
            check([os.path.join(self.dist_dir, whl_file)], strict=True)

    setup(
        packages=[],  # force skip the build-py step
        cmdclass={
            "bdist_wheel": BdistWheelWithCheck,
            "build_ext": CMakeBuild,
            "egg_info": EggInfo,
            "install_lib": CMakeInstall,
            "install_scripts": NoopInstallScript,
        },
        # build ext with cmake, so sources are not listed here
        ext_modules=[Extension(DistCfg.EXT_NAME, [])],
        options={
            # set build_lib for checking the existence of built modules
            "build_ext": {"build_lib": DistCfg.BUILD_DIR},
            "bdist_wheel": {"plat_name": DistCfg.PLAT},
        },
    )

else:
    from setuptools import find_packages

    setup(
        packages=find_packages(),
        package_data={
            "tensornet.core": ["_pywrap_tn.so"],
        },
        python_requires=">=3.7, <3.8",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.7",
        ],
    )
