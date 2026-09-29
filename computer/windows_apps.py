"""Trusted Windows application discovery from Start Menu and AppsFolder metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import time
from typing import Any

from computer.applications import (
    ApplicationCandidate, ApplicationMatch, match_applications, normalize_app_name,
)


_RESTRICTED_EXECUTABLES = frozenset({
    "cmd.exe", "powershell.exe", "pwsh.exe", "wt.exe", "regedit.exe", "reg.exe",
    "wscript.exe", "cscript.exe", "mshta.exe", "rundll32.exe", "wmic.exe", "mmc.exe",
    "taskschd.msc", "control.exe", "msiexec.exe", "python.exe", "pythonw.exe", "node.exe",
})
_RESTRICTED_NAMES = (
    "command prompt", "powershell", "windows terminal", "registry editor", "task scheduler",
    "computer management", "local security policy", "group policy", "windows script host",
    "developer command prompt", "developer powershell",
)


@dataclass(frozen=True, slots=True)
class ShortcutMetadata:
    target_path: str
    arguments: str = ""
    publisher: str = ""


@dataclass(frozen=True, slots=True)
class PackagedMetadata:
    display_name: str
    app_user_model_id: str
    publisher: str = ""
    package_family: str = ""
    process_names: tuple[str, ...] = ()
    launcher: Callable[[], None] | None = None


@dataclass(frozen=True, slots=True)
class _Binding:
    launcher: Callable[[], None]
    packaged_aumid: str | None = None


@dataclass(frozen=True, slots=True)
class PackagedLaunchDiagnosticResult:
    """Sanitized result of one isolated packaged-app launch request."""

    request_succeeded: bool
    request_elapsed_ms: int | None
    returned_pid_present: bool = False
    hresult: int | None = None
    failure_reason: str | None = None


def _safe_display_name(value: str) -> str:
    return re.sub(r"[\x00-\x1f\x7f]", "", value).strip()[:160]


def _local_id(source: str, reference: str) -> str:
    digest = sha256(f"{source}\0{reference.casefold()}".encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return f"app_{digest}"


def _restricted(name: str, target_identity: str) -> bool:
    normalized = normalize_app_name(name)
    executable = Path(target_identity).name.casefold()
    return (executable in _RESTRICTED_EXECUTABLES or Path(executable).suffix == ".msc"
            or any(term in normalized for term in _RESTRICTED_NAMES))


def _default_shortcut_reader(path: Path) -> ShortcutMetadata:
    from win32com.client import Dispatch
    shortcut = Dispatch("WScript.Shell").CreateShortcut(str(path))
    return ShortcutMetadata(str(shortcut.TargetPath or ""), str(shortcut.Arguments or ""))


def _default_shortcut_launcher(path: Path) -> None:
    # The path came from a trusted Start Menu root and is never supplied by Jev.
    os.startfile(str(path))  # type: ignore[attr-defined]


def _default_packaged_reader() -> Iterable[PackagedMetadata]:
    try:
        from win32com.client import Dispatch
        folder = Dispatch("Shell.Application").NameSpace("shell:AppsFolder")
        if folder is None:
            return ()
        records: list[PackagedMetadata] = []
        for item in folder.Items():
            name = str(getattr(item, "Name", "") or "")
            aumid = str(getattr(item, "Path", "") or "")
            if not name or not aumid:
                continue
            family = aumid.split("!", 1)[0] if "!" in aumid else ""
            records.append(PackagedMetadata(
                name, aumid, package_family=family,
                launcher=lambda app=item: app.InvokeVerb("open"),
            ))
        return records
    except Exception:
        return ()


class WindowsApplicationCatalog:
    """Catalog whose private bindings are created only from trusted Windows metadata."""

    def __init__(
        self, *, start_menu_roots: Sequence[Path] | None = None,
        shortcut_reader: Callable[[Path], ShortcutMetadata] = _default_shortcut_reader,
        shortcut_launcher: Callable[[Path], None] = _default_shortcut_launcher,
        packaged_reader: Callable[[], Iterable[PackagedMetadata]] = _default_packaged_reader,
        candidate_limit: int = 5,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if start_menu_roots is None:
            roots = []
            for variable in ("ProgramData", "APPDATA"):
                base = os.environ.get(variable)
                if base:
                    roots.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
            start_menu_roots = roots
        self._roots = tuple(path.resolve() for path in start_menu_roots)
        self._shortcut_reader = shortcut_reader
        self._shortcut_launcher = shortcut_launcher
        self._packaged_reader = packaged_reader
        self.candidate_limit = candidate_limit
        self._candidates: tuple[ApplicationCandidate, ...] | None = None
        self._by_id: dict[str, ApplicationCandidate] = {}
        self._bindings: dict[str, _Binding] = {}

    def discover(self) -> tuple[ApplicationCandidate, ...]:
        if self._candidates is not None:
            return self._candidates
        candidates: list[ApplicationCandidate] = []
        seen: set[tuple[str, str]] = set()
        for root in self._roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.lnk"), key=lambda item: str(item).casefold()):
                try:
                    resolved = path.resolve()
                    resolved.relative_to(root)
                    metadata = self._shortcut_reader(resolved)
                    name = _safe_display_name(resolved.stem)
                    target = metadata.target_path.strip()
                    target_name = Path(target).name
                    if not name or not target or Path(target).suffix.casefold() not in {".exe", ".msc"}:
                        continue
                    policy = "deny" if _restricted(name, target) else "allow"
                    app_id = _local_id("start_menu", str(resolved))
                    key = (normalize_app_name(name), target_name.casefold())
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = ApplicationCandidate(
                        app_id, name, "start_menu", _safe_display_name(metadata.publisher), policy,
                        (target_name,) if target_name else (), "",
                    )
                    candidates.append(candidate)
                    self._bindings[app_id] = _Binding(lambda shortcut=resolved: self._shortcut_launcher(shortcut))
                except Exception:
                    continue
        for metadata in self._packaged_reader():
            try:
                name = _safe_display_name(metadata.display_name)
                aumid = metadata.app_user_model_id.strip()
                if not name or not aumid or metadata.launcher is None:
                    continue
                policy = "deny" if _restricted(name, aumid) else "allow"
                app_id = _local_id("packaged", aumid)
                key = (normalize_app_name(name), metadata.package_family.casefold() or aumid.casefold())
                if key in seen:
                    continue
                seen.add(key)
                candidate = ApplicationCandidate(
                    app_id, name, "packaged", _safe_display_name(metadata.publisher), policy,
                    tuple(Path(item).name for item in metadata.process_names), metadata.package_family,
                )
                candidates.append(candidate)
                self._bindings[app_id] = _Binding(metadata.launcher, aumid)
            except Exception:
                continue
        candidates.sort(key=lambda candidate: (normalize_app_name(candidate.display_name), candidate.id))
        self._candidates = tuple(candidates)
        self._by_id = {candidate.id: candidate for candidate in candidates}
        return self._candidates

    def find(self, query: str, limit: int | None = None) -> tuple[ApplicationMatch, ...]:
        return match_applications(
            query, self.discover(), self.candidate_limit if limit is None else limit,
        )

    def resolve(self, app_id: str) -> ApplicationCandidate | None:
        self.discover()
        return self._by_id.get(app_id)

    def launch(self, app_id: str) -> None:
        candidate = self.resolve(app_id)
        binding = self._bindings.get(app_id)
        if candidate is None or binding is None or candidate.launch_policy != "allow":
            raise ValueError("Application ID is unavailable or not approved for launch.")
        binding.launcher()

    def launch_packaged_for_diagnostic(
        self,
        app_id: str,
        mechanism: str,
        *,
        activation_manager: Callable[[str], PackagedLaunchDiagnosticResult] | None = None,
    ) -> PackagedLaunchDiagnosticResult:
        """Run one launch mechanism using only this catalog's trusted binding.

        The AUMID is kept private and passed only to the Windows COM adapter.
        This method is for the manual A/B diagnostic and does not affect launch().
        """
        candidate = self.resolve(app_id)
        binding = self._bindings.get(app_id)
        if (candidate is None or binding is None or candidate.source != "packaged"
                or candidate.launch_policy != "allow" or not binding.packaged_aumid):
            raise ValueError("Application ID is unavailable for packaged activation diagnostics.")
        if mechanism == "shell":
            started = time.perf_counter()
            try:
                binding.launcher()
            except KeyboardInterrupt:
                raise
            except Exception:
                return PackagedLaunchDiagnosticResult(
                    False, max(0, round((time.perf_counter() - started) * 1000)),
                    failure_reason="launch_request_failed",
                )
            return PackagedLaunchDiagnosticResult(
                True, max(0, round((time.perf_counter() - started) * 1000)),
            )
        if mechanism == "activation-manager":
            if activation_manager is None:
                return PackagedLaunchDiagnosticResult(
                    False, None, failure_reason="activation_manager_unavailable",
                )
            return activation_manager(binding.packaged_aumid)
        raise ValueError("Unsupported packaged activation diagnostic mechanism.")

    def identify(self, process_name: str, package_family: str = "") -> str | None:
        process = Path(process_name).name.casefold()
        package = package_family.casefold()
        matches = [candidate.id for candidate in self.discover() if (
            package and candidate.package_family.casefold() == package
        ) or (process and any(Path(name).name.casefold() == process for name in candidate.process_names))]
        return matches[0] if len(matches) == 1 else None
