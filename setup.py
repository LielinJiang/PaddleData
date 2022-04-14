import os
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from paddle.utils.cpp_extension import BuildExtension as PaddleBuildExtension


def _get_files(path):
    """
    Helps to list all files under the given path.
    """
    if os.path.isfile(path):
        return [path]
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        for file in files:
            file = os.path.join(root, file)
            all_files.append(file)
    return all_files

class BuildExtension(PaddleBuildExtension):
    """
    Support both `CppExtention` of Paddle and custom extensions of PaddleNLP.
    """

    def build_extensions(self):
        custom_exts = []  # for
        no_custom_exts = []  # for normal extentions paddle.utils.cpp_extension
        for ext in self.extensions:
            if hasattr(ext, "build_with_command"):
                # custom build in Extension
                ext.build_with_command(self)
                custom_exts.append(ext)
            else:
                no_custom_exts.append(ext)
        if no_custom_exts:
            # Build CppExtentio/CUDAExtension with `PaddleBuildExtension`
            self.extensions = no_custom_exts
            super(BuildExtension, self).build_extensions()
        self.extensions = custom_exts + no_custom_exts


class PaddleDataExtension(Extension):
    def __init__(self, name, source_dir=None):
        # A CMakeExtension needs a source_dir instead of a file list.
        Extension.__init__(self, name, sources=[])
        if source_dir is None:
            self.source_dir = str(Path(__file__).parent.resolve())
        else:
            self.source_dir = os.path.abspath(os.path.expanduser(source_dir))
        # self.sources = _get_files(self.source_dir)
        self.sources = _get_files(
            os.path.
            join(self.source_dir, "paddledata", "decode"))
        self._std_out_handle = None

    def build_with_command(self, ext_builder):
        """
        Custom `build_ext.build_extension` in `Extension` instead of `Command`.
        `ext_builder` is the instance of `build_ext` command.
        """
        # refer to https://github.com/pybind/cmake_example/blob/master/setup.py
        if ext_builder.compiler.compiler_type == "msvc":
            raise NotImplementedError
        cmake_args = getattr(self, "cmake_args", []) + [
            "-DCMAKE_BUILD_TYPE={}".format("Debug"
                                           if ext_builder.debug else "Release"),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_builder.build_lib),
        ]
        build_args = []

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(ext_builder, "parallel") and ext_builder.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(16)]
                # build_args += ["-j{}".format(ext_builder.parallel)]

        if not os.path.exists(ext_builder.build_temp):
            os.makedirs(ext_builder.build_temp)

        # Redirect stdout/stderr to mute, especially when allowing errors
        stdout = getattr(self, "_std_out_handle", None)
        print("cmake args:", cmake_args)
        subprocess.check_call(
            ["cmake", self.source_dir] + cmake_args,
            cwd=ext_builder.build_temp,
            stdout=stdout,
            stderr=stdout)
        print('finish00')
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=ext_builder.build_temp,
            stdout=stdout,
            stderr=stdout)
        print('finish01')
        print(ext_builder.build_temp)
        print(ext_builder.build_lib)
        ext_builder.copy_tree(
                os.path.join(ext_builder.build_temp, "paddledata", "decode", "libs"),
                ext_builder.build_lib)

    def get_target_filename(self):
        return "libimage_decode_op.so"

setup(
    name='PaddleData',
    ext_modules=[
        PaddleDataExtension("libimage_decode_op", "./")
        ],
    cmdclass={'build_ext' : BuildExtension.with_options(
        output_dir=r'libs')
    })
