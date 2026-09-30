"""Trusted Windows application discovery from Start Menu and AppsFolder metadata."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
import re
import time
from typing import Any

from computer.applications import (
    ApplicationCandidate, ApplicationIdentityComparisonDiagnostics,
    ApplicationIdentityEquivalence, ApplicationIdentityEvidence, ApplicationMatch,
    compare_application_identity_evidence, group_application_candidates, match_applications,
    normalize_app_identity, normalize_app_name, safe_application_id,
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
    app_user_model_id: str = ""
    product_name: str = ""


@dataclass(frozen=True, slots=True)
class PackagedMetadata:
    display_name: str
    app_user_model_id: str
    publisher: str = ""
    package_family: str = ""
    process_names: tuple[str, ...] = ()
    launcher: Callable[[], None] | None = None
    executable_path: str = ""
    product_name: str = ""


@dataclass(frozen=True, slots=True)
class _Binding:
    launcher: Callable[[], None]
    packaged_aumid: str | None = None
    identity_target_path: str | None = None


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


def _identity_fingerprint(namespace: str, identity: str) -> str:
    digest = sha256(f"{namespace}\0{identity}".encode("utf-8", "surrogatepass")).hexdigest()
    return f"{namespace}:{digest}"


def _value_fingerprint(
    namespace: str, value: str, *, case_insensitive: bool = True,
) -> str | None:
    normalized = value.strip()
    if not normalized:
        return None
    identity = normalized.casefold() if case_insensitive else normalized
    return _identity_fingerprint(namespace, identity)


def _shortcut_app_user_model_id(path: Path) -> str:
    try:
        from win32com.propsys import propsys, pscon
        store = propsys.SHGetPropertyStoreFromParsingName(
            str(path), None, 0, propsys.IID_IPropertyStore,
        )
        value = store.GetValue(pscon.PKEY_AppUserModel_ID)
        raw_value = value.GetValue() if hasattr(value, "GetValue") else value
        if isinstance(raw_value, str) and 0 < len(raw_value) <= 512:
            return raw_value.strip()
    except Exception:
        return ""
    return ""


def _version_product_metadata(path: str) -> tuple[str, str]:
    """Read optional executable company/product strings without launching it."""
    try:
        import win32api
        translations = win32api.GetFileVersionInfo(path, r"\VarFileInfo\Translation")
        for language, codepage in translations:
            table = f"\\StringFileInfo\\{language:04x}{codepage:04x}\\"
            company = str(win32api.GetFileVersionInfo(path, table + "CompanyName") or "").strip()
            product = str(win32api.GetFileVersionInfo(path, table + "ProductName") or "").strip()
            if company or product:
                return company[:160], product[:160]
        return "", ""
    except Exception:
        return "", ""


def _shortcut_identity(target: str, arguments: str) -> ApplicationIdentityEvidence:
    expanded = os.path.expandvars(target.strip())
    target_path = Path(expanded)
    if not target_path.is_absolute():
        return ApplicationIdentityEvidence(
            shortcut_arguments_fingerprint=_value_fingerprint(
                "arguments", arguments, case_insensitive=False,
            ),
        )
    try:
        canonical_target = os.path.normcase(str(target_path.resolve(strict=False)))
    except (OSError, RuntimeError):
        return ApplicationIdentityEvidence(
            shortcut_arguments_fingerprint=_value_fingerprint(
                "arguments", arguments, case_insensitive=False,
            ),
        )
    # Arguments are part of the trusted shortcut's launch identity. They are
    # hashed with the target and never included in candidate diagnostics.
    catalog_identity = _identity_fingerprint(
        "shortcut", f"{canonical_target.casefold()}\0{arguments.strip()}",
    )
    executable_identity = (
        _value_fingerprint("executable", canonical_target)
        if target_path.suffix.casefold() == ".exe" else None
    )
    return ApplicationIdentityEvidence(
        executable_identity_fingerprint=executable_identity,
        shortcut_arguments_fingerprint=_identity_fingerprint("arguments", arguments),
        catalog_identity_fingerprint=catalog_identity,
    )


def _app_user_model_id_evidence(value: str) -> tuple[str | None, str | None]:
    app_id = value.strip()
    if not app_id:
        return None, None
    app_id_fingerprint = _value_fingerprint("app_user_model_id", app_id)
    family = app_id.split("!", 1)[0] if "!" in app_id else ""
    return app_id_fingerprint, _value_fingerprint("package_identity", family)


def _restricted(name: str, target_identity: str) -> bool:
    normalized = normalize_app_name(name)
    executable = Path(target_identity).name.casefold()
    return (executable in _RESTRICTED_EXECUTABLES or Path(executable).suffix == ".msc"
            or any(term in normalized for term in _RESTRICTED_NAMES))


def _default_shortcut_reader(path: Path) -> ShortcutMetadata:
    from win32com.client import Dispatch
    shortcut = Dispatch("WScript.Shell").CreateShortcut(str(path))
    target = str(shortcut.TargetPath or "")
    return ShortcutMetadata(
        target, str(shortcut.Arguments or ""), app_user_model_id=_shortcut_app_user_model_id(path),
    )


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
        self._logical_members: dict[str, tuple[ApplicationCandidate, ...]] = {}
        self._logical_by_id: dict[str, ApplicationCandidate] = {}
        self._logical_id_by_member_id: dict[str, str] = {}
        self._identity_evidence_by_id: dict[str, ApplicationIdentityEvidence] = {}

    def discover(self) -> tuple[ApplicationCandidate, ...]:
        if self._candidates is not None:
            return self._candidates
        candidates: list[ApplicationCandidate] = []
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
                    identity_evidence = _shortcut_identity(target, metadata.arguments)
                    app_user_model_id = (
                        metadata.app_user_model_id or _shortcut_app_user_model_id(resolved)
                    )
                    app_user_model_id_fingerprint, package_identity_fingerprint = (
                        _app_user_model_id_evidence(app_user_model_id)
                    )
                    publisher_product_fingerprint = (
                        _value_fingerprint(
                            "publisher_product",
                            f"{metadata.publisher}\0{metadata.product_name}",
                        ) if metadata.publisher or metadata.product_name else
                        identity_evidence.publisher_product_fingerprint
                    )
                    identity_evidence = replace(
                        identity_evidence,
                        app_user_model_id_fingerprint=app_user_model_id_fingerprint,
                        package_identity_fingerprint=package_identity_fingerprint,
                        publisher_product_fingerprint=publisher_product_fingerprint,
                    )
                    candidate = ApplicationCandidate(
                        app_id, name, "start_menu", _safe_display_name(metadata.publisher), policy,
                        (target_name,) if target_name else (), "",
                        identity_fingerprint=identity_evidence.catalog_identity_fingerprint,
                        launch_target_type="shortcut",
                    )
                    candidates.append(candidate)
                    self._identity_evidence_by_id[app_id] = identity_evidence
                    self._bindings[app_id] = _Binding(
                        lambda shortcut=resolved: self._shortcut_launcher(shortcut),
                        identity_target_path=target,
                    )
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
                app_user_model_id_fingerprint, aumid_package_fingerprint = (
                    _app_user_model_id_evidence(aumid)
                )
                package_identity_fingerprint = _value_fingerprint(
                    "package_identity", metadata.package_family,
                ) or aumid_package_fingerprint
                executable_evidence = (
                    _shortcut_identity(metadata.executable_path, "")
                    if metadata.executable_path else ApplicationIdentityEvidence()
                )
                product_fingerprint = _value_fingerprint(
                    "publisher_product", f"{metadata.publisher}\0{metadata.product_name}",
                ) if metadata.publisher or metadata.product_name else None
                identity_evidence = ApplicationIdentityEvidence(
                    app_user_model_id_fingerprint=app_user_model_id_fingerprint,
                    package_identity_fingerprint=package_identity_fingerprint,
                    executable_identity_fingerprint=executable_evidence.executable_identity_fingerprint,
                    publisher_product_fingerprint=product_fingerprint,
                    catalog_identity_fingerprint=_identity_fingerprint("packaged", aumid.casefold()),
                )
                candidate = ApplicationCandidate(
                    app_id, name, "packaged", _safe_display_name(metadata.publisher), policy,
                    tuple(Path(item).name for item in metadata.process_names), metadata.package_family,
                    identity_fingerprint=identity_evidence.catalog_identity_fingerprint,
                    launch_target_type="packaged",
                )
                candidates.append(candidate)
                self._identity_evidence_by_id[app_id] = identity_evidence
                self._bindings[app_id] = _Binding(
                    metadata.launcher, aumid,
                    metadata.executable_path or None,
                )
            except Exception:
                continue
        candidates = self._apply_proven_equivalence(candidates)
        candidates.sort(key=lambda candidate: (normalize_app_identity(candidate.display_name), candidate.id))
        self._candidates = tuple(candidates)
        self._by_id = {candidate.id: candidate for candidate in candidates}
        groups = group_application_candidates(self._candidates)
        for logical, members in groups:
            self._logical_by_id[logical.id] = logical
            self._logical_members[logical.id] = members
            for member in members:
                self._logical_id_by_member_id[member.id] = logical.id
        self._by_id.update(self._logical_by_id)
        return self._candidates

    def _apply_proven_equivalence(
        self, candidates: list[ApplicationCandidate],
    ) -> list[ApplicationCandidate]:
        parents = list(range(len(candidates)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        comparisons: dict[tuple[int, int], ApplicationIdentityEquivalence] = {}
        for left_index, left in enumerate(candidates):
            left_evidence = self._identity_evidence_by_id.get(left.id)
            for right_index in range(left_index + 1, len(candidates)):
                right = candidates[right_index]
                result = compare_application_identity_evidence(
                    left_evidence, self._identity_evidence_by_id.get(right.id),
                )
                comparisons[(left_index, right_index)] = result
                if result.equivalence_result == "PROVEN_EQUIVALENT":
                    left_root, right_root = find(left_index), find(right_index)
                    if left_root != right_root:
                        parents[right_root] = left_root

        components: dict[int, list[int]] = {}
        for index in range(len(candidates)):
            components.setdefault(find(index), []).append(index)
        replacements: dict[int, ApplicationCandidate] = {}
        for members in components.values():
            if len(members) < 2:
                continue
            # Equivalence is not assumed transitive across partial metadata.
            # Collapse only a clique of pairwise proven identities; an UNKNOWN
            # pair must keep the component ambiguous even if another path linked it.
            not_fully_proven = any(
                comparisons[(min(left, right), max(left, right))].equivalence_result
                != "PROVEN_EQUIVALENT"
                for offset, left in enumerate(members)
                for right in members[offset + 1:]
            )
            if not_fully_proven:
                continue
            group_identity = _identity_fingerprint(
                "proven-equivalence-group",
                "\0".join(sorted(candidates[index].id for index in members)),
            )
            for index in members:
                replacements[index] = replace(
                    candidates[index], identity_fingerprint=group_identity,
                )
        return [replacements.get(index, candidate) for index, candidate in enumerate(candidates)]

    def find(self, query: str, limit: int | None = None) -> tuple[ApplicationMatch, ...]:
        return match_applications(
            query, self.discover(), self.candidate_limit if limit is None else limit,
        )

    def compare_identities(
        self, left_app_id: str, right_app_id: str,
    ) -> ApplicationIdentityComparisonDiagnostics:
        """Compare two catalog IDs without revealing their underlying identity values."""
        self.discover()
        left_evidence = self._identity_evidence_for_audit(left_app_id)
        right_evidence = self._identity_evidence_for_audit(right_app_id)
        result = compare_application_identity_evidence(
            left_evidence, right_evidence,
        )
        return ApplicationIdentityComparisonDiagnostics(
            left_app_id=safe_application_id(left_app_id),
            right_app_id=safe_application_id(right_app_id),
            comparison_attempted=result.comparison_attempted,
            equivalence_result=result.equivalence_result,
            equivalence_signal_kind=result.equivalence_signal_kind,
            compared_identity_fields_present=result.compared_identity_fields_present,
        )

    def _identity_evidence_for_audit(
        self, app_id: str,
    ) -> ApplicationIdentityEvidence | None:
        evidence = self._identity_evidence_by_id.get(app_id)
        binding = self._bindings.get(app_id)
        if evidence is None or binding is None or evidence.publisher_product_fingerprint:
            return evidence
        target = binding.identity_target_path
        if not target or not Path(target).is_file():
            return evidence
        company, product = _version_product_metadata(target)
        if not company and not product:
            return evidence
        enriched = replace(
            evidence,
            publisher_product_fingerprint=_value_fingerprint(
                "publisher_product", f"{company}\0{product}",
            ),
        )
        self._identity_evidence_by_id[app_id] = enriched
        return enriched

    def resolve(self, app_id: str) -> ApplicationCandidate | None:
        self.discover()
        return self._by_id.get(app_id)

    def launch(self, app_id: str) -> None:
        candidate = self.resolve(app_id)
        members = self._logical_members.get(app_id, ())
        binding = next((self._bindings[item.id] for item in members if item.id in self._bindings), None)
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
        matches = [candidate for candidate in self.discover() if (
            package and candidate.package_family.casefold() == package
        ) or (process and any(Path(name).name.casefold() == process for name in candidate.process_names))]
        logical_ids = {self._logical_id_by_member_id.get(candidate.id, candidate.id) for candidate in matches}
        return next(iter(logical_ids)) if len(logical_ids) == 1 else None
