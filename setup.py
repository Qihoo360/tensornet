import os, subprocess, sys

from importlib.machinery import SourceFileLoader

# use importlib to avoid import so file
_version = SourceFileLoader('version', 'tensornet/version.py').load_module()

class DistCfg():
    VER = _version.VERSION
    EXT_NAME = 'tensornet.core._pywrap_tn'
    PLAT = "manylinux2010_x86_64"
    BUILD_PRESET = "release"
    BUILD_DIR = "build/release"

if "PIXI_IN_SHELL" in os.environ:
    from setuptools import setup, Extension
    from setuptools.command.bdist_wheel import bdist_wheel
    from setuptools.command.build_ext import build_ext
    from setuptools.command.egg_info import egg_info
    from setuptools.command.install_lib import install_lib
    from setuptools.command.install_scripts import install_scripts
    from twine.commands.check import check

    class CMakeBuild(build_ext):
        def build_extensions(self):
            subprocess.check_call([ 'cmake', '--workflow', '--preset', DistCfg.BUILD_PRESET, '--fresh' ])
            # Make sure CMake generated all the appropriate extensions
            for ext in self.extensions:
                ext = self.get_ext_fullpath(ext.name)
                if not os.path.exists(ext):
                    raise RuntimeError(f'Failed to build extension: {ext}')

    class CMakeInstall(install_lib):
        def install(self):
            subprocess.check_call([
                'cmake', '--build',
                '--preset', DistCfg.BUILD_PRESET, '--target', 'install',
                '--', "SKIP_RUN_TESTS=1" ])

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
        name = 'qihoo-tensornet',
        version = DistCfg.VER,
        description = 'tensornet',
        long_description = open('README.md').read(),
        long_description_content_type = 'text/markdown',
        author = 'jiangxinglei',
        author_email = 'jiangxinglei@360.cn',
        url = 'https://github.com/Qihoo360/tensornet',
        python_requires = '>=3.6, <3.9',
        classifiers = [
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
        ],
        platforms = [DistCfg.PLAT],

        cmdclass = {
            "bdist_wheel": BdistWheelWithCheck,
            'build_ext': CMakeBuild,
            'egg_info': EggInfo,
            'install_lib': CMakeInstall,
            'install_scripts': NoopInstallScript ,
        },

        # build ext with cmake, so sources are not listed here
        ext_modules=[Extension(DistCfg.EXT_NAME, [])],

        # force skip the build-py step
        packages=[],

        options = {
            # set build_lib for checking the existence of built modules
            "build_ext": { "build_lib": DistCfg.BUILD_DIR },
            'bdist_wheel': { "plat_name": DistCfg.PLAT },
        },
    )

else:
    from setuptools import setup, find_packages
    setup(
        name='qihoo-tensornet',
        version = DistCfg.VER,
        description='tensornet',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        author='jiangxinglei',
        author_email='jiangxinglei@360.cn',
        url='https://github.com/Qihoo360/tensornet',
        packages=find_packages(),
        package_data = {
            "tensornet.core": ["_pywrap_tn.so"],
        },
        python_requires='>=3.7, <3.8',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.7'
        ],
        platforms = [DistCfg.PLAT],
    )
