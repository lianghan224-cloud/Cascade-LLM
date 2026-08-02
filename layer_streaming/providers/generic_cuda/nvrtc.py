"""Minimal NVRTC and CUDA Driver wrappers for a compiler-free runtime image."""

import ctypes
from pathlib import Path
import threading


class NVRTCError(RuntimeError):
    pass


def _load_nvrtc():
    candidates = []
    try:
        import nvidia.cuda_nvrtc

        candidates.append(
            Path(nvidia.cuda_nvrtc.__file__).resolve().parent
            / "lib/libnvrtc.so.12"
        )
    except ImportError:
        pass
    candidates.extend((Path("libnvrtc.so.12"), Path("libnvrtc.so")))
    errors = []
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
        except OSError as error:
            errors.append(str(error))
    raise NVRTCError("NVRTC library could not be loaded: {}".format(errors))


def _check_nvrtc(library, status, operation, program=None):
    if int(status) == 0:
        return
    library.nvrtcGetErrorString.argtypes = [ctypes.c_int]
    library.nvrtcGetErrorString.restype = ctypes.c_char_p
    message = library.nvrtcGetErrorString(int(status))
    log = ""
    if program is not None:
        size = ctypes.c_size_t()
        if library.nvrtcGetProgramLogSize(program, ctypes.byref(size)) == 0 and size.value:
            buffer = ctypes.create_string_buffer(size.value)
            if library.nvrtcGetProgramLog(program, buffer) == 0:
                log = buffer.value.decode("utf-8", errors="replace")
    raise NVRTCError(
        "{} failed: {}{}".format(
            operation,
            message.decode("utf-8", errors="replace") if message else status,
            "\n" + log if log else "",
        )
    )


class CUDAKernelModule:
    """Compile CUDA C to PTX with NVRTC and launch through libcuda."""

    def __init__(self, source, architecture, functions):
        self.source = str(source)
        self.architecture = str(architecture)
        self.function_names = tuple(functions)
        self._nvrtc = _load_nvrtc()
        self._cuda = ctypes.CDLL("libcuda.so.1")
        self._lock = threading.RLock()
        self._module = ctypes.c_void_p()
        self._functions = {}
        self._configure_apis()
        ptx = self._compile()
        self._load(ptx)

    def _configure_apis(self):
        nvrtc = self._nvrtc
        nvrtc.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        nvrtc.nvrtcCreateProgram.restype = ctypes.c_int
        nvrtc.nvrtcCompileProgram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        nvrtc.nvrtcCompileProgram.restype = ctypes.c_int
        nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        nvrtc.nvrtcGetPTXSize.restype = ctypes.c_int
        nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nvrtc.nvrtcGetPTX.restype = ctypes.c_int
        nvrtc.nvrtcGetProgramLogSize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        nvrtc.nvrtcGetProgramLogSize.restype = ctypes.c_int
        nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nvrtc.nvrtcGetProgramLog.restype = ctypes.c_int
        nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        nvrtc.nvrtcDestroyProgram.restype = ctypes.c_int

        cuda = self._cuda
        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        cuda.cuModuleLoadData.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        cuda.cuModuleLoadData.restype = ctypes.c_int
        cuda.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        cuda.cuModuleGetFunction.restype = ctypes.c_int
        cuda.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        cuda.cuLaunchKernel.restype = ctypes.c_int
        cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
        cuda.cuModuleUnload.restype = ctypes.c_int
        cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        cuda.cuGetErrorName.restype = ctypes.c_int
        cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        cuda.cuGetErrorString.restype = ctypes.c_int

    def _check_cuda(self, status, operation):
        if int(status) == 0:
            return
        name = ctypes.c_char_p()
        message = ctypes.c_char_p()
        self._cuda.cuGetErrorName(int(status), ctypes.byref(name))
        self._cuda.cuGetErrorString(int(status), ctypes.byref(message))
        raise NVRTCError(
            "{} failed: {} ({})".format(
                operation,
                message.value.decode("utf-8", errors="replace") if message.value else status,
                name.value.decode("ascii", errors="replace") if name.value else "unknown",
            )
        )

    def _compile(self):
        program = ctypes.c_void_p()
        source = self.source.encode("utf-8")
        status = self._nvrtc.nvrtcCreateProgram(
            ctypes.byref(program),
            source,
            b"cascade_paged_attention.cu",
            0,
            None,
            None,
        )
        _check_nvrtc(self._nvrtc, status, "nvrtcCreateProgram")
        try:
            options = (
                "--gpu-architecture=compute_{}".format(
                    self.architecture.replace("sm", "")
                ).encode("ascii"),
                b"--std=c++14",
                b"--device-as-default-execution-space",
            )
            option_array = (ctypes.c_char_p * len(options))(*options)
            status = self._nvrtc.nvrtcCompileProgram(
                program,
                len(options),
                option_array,
            )
            _check_nvrtc(
                self._nvrtc,
                status,
                "nvrtcCompileProgram",
                program=program,
            )
            size = ctypes.c_size_t()
            _check_nvrtc(
                self._nvrtc,
                self._nvrtc.nvrtcGetPTXSize(program, ctypes.byref(size)),
                "nvrtcGetPTXSize",
            )
            buffer = ctypes.create_string_buffer(size.value)
            _check_nvrtc(
                self._nvrtc,
                self._nvrtc.nvrtcGetPTX(program, buffer),
                "nvrtcGetPTX",
            )
            return bytes(buffer.raw)
        finally:
            self._nvrtc.nvrtcDestroyProgram(ctypes.byref(program))

    def _load(self, ptx):
        self._check_cuda(self._cuda.cuInit(0), "cuInit")
        buffer = ctypes.create_string_buffer(ptx)
        self._ptx_buffer = buffer
        self._check_cuda(
            self._cuda.cuModuleLoadData(
                ctypes.byref(self._module),
                ctypes.cast(buffer, ctypes.c_void_p),
            ),
            "cuModuleLoadData",
        )
        for name in self.function_names:
            function = ctypes.c_void_p()
            self._check_cuda(
                self._cuda.cuModuleGetFunction(
                    ctypes.byref(function),
                    self._module,
                    name.encode("ascii"),
                ),
                "cuModuleGetFunction({})".format(name),
            )
            self._functions[name] = function

    def launch(self, name, grid, block, stream, arguments, shared_memory=0):
        try:
            function = self._functions[name]
        except KeyError:
            raise KeyError("unknown CUDA kernel {}".format(name))
        keepalive = []
        argument_pointers = []
        for ctype, value in arguments:
            scalar = ctype(value)
            keepalive.append(scalar)
            argument_pointers.append(ctypes.c_void_p(ctypes.addressof(scalar)))
        parameter_array = (ctypes.c_void_p * len(argument_pointers))(
            *argument_pointers
        )
        with self._lock:
            self._check_cuda(
                self._cuda.cuLaunchKernel(
                    function,
                    int(grid[0]),
                    int(grid[1]),
                    int(grid[2]),
                    int(block[0]),
                    int(block[1]),
                    int(block[2]),
                    int(shared_memory),
                    ctypes.c_void_p(int(stream)),
                    parameter_array,
                    None,
                ),
                "cuLaunchKernel({})".format(name),
            )

    def close(self):
        with self._lock:
            if self._module:
                self._check_cuda(self._cuda.cuModuleUnload(self._module), "cuModuleUnload")
                self._module = ctypes.c_void_p()
                self._functions.clear()
