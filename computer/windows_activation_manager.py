"""Isolated ctypes adapter for IApplicationActivationManager diagnostics."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import threading
import time
from typing import Protocol
from uuid import UUID

from computer.windows_apps import PackagedLaunchDiagnosticResult


_CLSID_APPLICATION_ACTIVATION_MANAGER = "45BA127D-10A8-46EA-8AB7-56EA9078943C"
_IID_IAPPLICATION_ACTIVATION_MANAGER = "2E941141-7F97-4756-BA1D-9DECDE894A3D"
_CLSCTX_LOCAL_SERVER = 0x4
_COINIT_MULTITHREADED = 0x0
_COINIT_DISABLE_OLE1DDE = 0x4
_COM_INITIALIZATION_FLAGS = _COINIT_MULTITHREADED | _COINIT_DISABLE_OLE1DDE
_ACTIVATE_APPLICATION_VTABLE_INDEX = 3  # IUnknown's three methods precede it.


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(value: str) -> _GUID:
    parsed = UUID(value)
    return _GUID(
        parsed.time_low, parsed.time_mid, parsed.time_hi_version,
        (ctypes.c_ubyte * 8).from_buffer_copy(parsed.bytes[8:]),
    )


class _ComFailure(Exception):
    def __init__(self, hresult: int) -> None:
        super().__init__()
        self.hresult = hresult


class _ComRuntime(Protocol):
    def initialize_com(self) -> None: ...
    def create_manager(self) -> object: ...
    def activate_application(self, manager: object, app_user_model_id: str) -> tuple[int, int]: ...
    def release_manager(self, manager: object) -> None: ...
    def uninitialize(self) -> None: ...


class _WindowsComRuntime:
    """Small raw COM vtable adapter; all COM calls stay on its STA worker."""

    def __init__(self) -> None:
        self._ole32 = ctypes.WinDLL("ole32", use_last_error=True)
        self._co_initialize_ex = self._ole32.CoInitializeEx
        self._co_initialize_ex.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        self._co_initialize_ex.restype = ctypes.c_long
        self._co_create_instance = self._ole32.CoCreateInstance
        self._co_create_instance.argtypes = (
            ctypes.POINTER(_GUID), ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p),
        )
        self._co_create_instance.restype = ctypes.c_long
        self._co_uninitialize = self._ole32.CoUninitialize
        self._co_uninitialize.argtypes = ()
        self._co_uninitialize.restype = None
        self._winfunctype = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)

    def initialize_com(self) -> None:
        hresult = int(self._co_initialize_ex(None, _COM_INITIALIZATION_FLAGS))
        if hresult < 0:
            raise _ComFailure(hresult)

    def create_manager(self) -> object:
        clsid = _guid(_CLSID_APPLICATION_ACTIVATION_MANAGER)
        iid = _guid(_IID_IAPPLICATION_ACTIVATION_MANAGER)
        manager = ctypes.c_void_p()
        hresult = int(self._co_create_instance(
            ctypes.byref(clsid), None, _CLSCTX_LOCAL_SERVER,
            ctypes.byref(iid), ctypes.byref(manager),
        ))
        if hresult < 0:
            raise _ComFailure(hresult)
        if not manager.value:
            raise _ComFailure(-2147467261)  # E_POINTER
        return manager

    def activate_application(
        self, manager: object, app_user_model_id: str,
    ) -> tuple[int, int]:
        pointer = ctypes.cast(manager, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
        vtable = pointer.contents
        method_type = self._winfunctype(
            ctypes.c_long, ctypes.c_void_p, ctypes.c_wchar_p,
            ctypes.c_wchar_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
        )
        activate = method_type(vtable[_ACTIVATE_APPLICATION_VTABLE_INDEX])
        process_id = ctypes.c_uint32()
        hresult = int(activate(
            manager, app_user_model_id, None, 0, ctypes.byref(process_id),
        ))
        return hresult, int(process_id.value)

    def release_manager(self, manager: object) -> None:
        pointer = ctypes.cast(manager, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))
        release_type = self._winfunctype(ctypes.c_uint32, ctypes.c_void_p)
        release = release_type(pointer.contents[2])
        release(manager)

    def uninitialize(self) -> None:
        self._co_uninitialize()


def _safe_hresult(error: Exception) -> int | None:
    value = getattr(error, "hresult", None)
    if not isinstance(value, int):
        return None
    return value & 0xFFFFFFFF


def activate_application(
    app_user_model_id: str,
    *,
    runtime_factory: Callable[[], _ComRuntime] = _WindowsComRuntime,
    clock: Callable[[], float] = time.perf_counter,
) -> PackagedLaunchDiagnosticResult:
    """Call ActivateApplication with the catalog AUMID, no args, and AO_NONE."""
    try:
        runtime = runtime_factory()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return PackagedLaunchDiagnosticResult(
            False, None, hresult=_safe_hresult(exc),
            failure_reason="com_initialization_failed",
        )

    results: list[PackagedLaunchDiagnosticResult] = []

    def worker() -> None:
        initialized = False
        manager: object | None = None
        stage = "com_initialization_failed"
        try:
            runtime.initialize_com()
            initialized = True
            stage = "manager_creation_failed"
            manager = runtime.create_manager()
            stage = "activation_failed"
            started = clock()
            try:
                hresult, process_id = runtime.activate_application(manager, app_user_model_id)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                results.append(PackagedLaunchDiagnosticResult(
                    False, max(0, round((clock() - started) * 1000)),
                    hresult=_safe_hresult(exc), failure_reason=stage,
                ))
            else:
                elapsed_ms = max(0, round((clock() - started) * 1000))
                if hresult < 0:
                    results.append(PackagedLaunchDiagnosticResult(
                        False, elapsed_ms, hresult=hresult & 0xFFFFFFFF,
                        failure_reason=stage,
                    ))
                else:
                    results.append(PackagedLaunchDiagnosticResult(
                        True, elapsed_ms, returned_pid_present=process_id > 0,
                        hresult=hresult & 0xFFFFFFFF,
                    ))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            results.append(PackagedLaunchDiagnosticResult(
                False, None, hresult=_safe_hresult(exc), failure_reason=stage,
            ))
        finally:
            if manager is not None:
                try:
                    runtime.release_manager(manager)
                except Exception:
                    pass
            if initialized:
                try:
                    runtime.uninitialize()
                except Exception:
                    pass
            if not results:
                results.append(PackagedLaunchDiagnosticResult(
                    False, None, failure_reason="activation_failed",
                ))

    thread = threading.Thread(
        target=worker, name="voice-jev-packaged-activation", daemon=True,
    )
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        raise
    return results[0]


__all__ = ["activate_application"]
