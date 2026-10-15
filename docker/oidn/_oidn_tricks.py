"""Replacement for sapien/_oidn_tricks.py (SAPIEN 3.0.0b1) that preloads Intel Open Image Denoise 2.3.3.

SAPIEN 3.0.0b1 bundles OIDN 2.0.1, which predates NVIDIA Blackwell. On RTX PRO 6000
every denoise call fails with "[svulkan2] [error] OIDN Error: invalid handle" and the
ray-traced camera images reach the policy un-denoised. OIDN 2.3.3 added Blackwell
support. The launcher binds this file over the installed copy and binds a directory
holding the 2.3.3 libraries over sapien/oidn_library/. libsvulkan2 links the soname
libOpenImageDenoise.so.2, which the preload below satisfies.
"""
import ctypes
import platform
import os

if platform.system() == "Linux":
    try:
        oidn_dll = ctypes.CDLL(
            os.path.join(os.path.dirname(__file__), "oidn_library/libOpenImageDenoise_core.so.2.3.3"),
            ctypes.RTLD_LOCAL,
        )
        oidn_dll = ctypes.CDLL(
            os.path.join(os.path.dirname(__file__), "oidn_library/libOpenImageDenoise.so.2.3.3"),
            ctypes.RTLD_LOCAL,
        )
    except Exception:
        pass
