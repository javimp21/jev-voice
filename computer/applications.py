"""Platform-neutral trusted application catalog contracts and matching."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import re
from typing import Literal, Protocol
import unicodedata


LaunchPolicy = Literal["allow", "confirm", "deny"]
ApplicationSource = Literal["start_menu", "packaged", "test"]


@dataclass(frozen=True, slots=True)
class ApplicationCandidate:
    """Sanitized public identity; trusted launch references remain catalog-private."""

    id: str
    display_name: str
    source: ApplicationSource
    publisher: str = ""
    launch_policy: LaunchPolicy = "confirm"
    process_names: tuple[str, ...] = ()
    package_family: str = ""


@dataclass(frozen=True, slots=True)
class ApplicationMatch:
    candidate: ApplicationCandidate
    score: int


class ApplicationCatalog(Protocol):
    def discover(self) -> tuple[ApplicationCandidate, ...]: ...
    def find(self, query: str, limit: int = 5) -> tuple[ApplicationMatch, ...]: ...
    def resolve(self, app_id: str) -> ApplicationCandidate | None: ...
    def launch(self, app_id: str) -> None: ...
    def identify(self, process_name: str, package_family: str = "") -> str | None: ...


def normalize_app_name(value: str) -> str:
    # Decorative symbols (for example the trademark sign) carry no useful
    # identity signal. Remove them before NFKC can expand them into letters.
    without_symbols = "".join(character for character in value
                              if not unicodedata.category(character).startswith("S"))
    normalized = unicodedata.normalize("NFKC", without_symbols).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


_QUERY_STOPWORDS = frozenset({
    "open", "launch", "start", "run", "and", "then", "search", "for", "go", "to", "the",
    "app", "application", "write", "type", "enter", "find", "navigate", "in", "with", "a", "an",
})


def _score(query: str, candidate: ApplicationCandidate) -> int:
    query_name = normalize_app_name(query)
    name = normalize_app_name(candidate.display_name)
    if not query_name or not name:
        return 0
    if query_name == name:
        return 100
    query_tokens = set(query_name.split()) - _QUERY_STOPWORDS
    name_tokens = set(name.split())
    if not query_tokens or not name_tokens:
        return 0
    if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", query_name):
        return 95
    overlap = query_tokens & name_tokens
    if not overlap:
        return 0
    if name_tokens <= query_tokens:
        return 90
    if query_tokens <= name_tokens:
        return 85
    coverage = len(overlap) / len(name_tokens)
    return 70 + round(15 * coverage) if coverage >= 0.5 else 0


def match_applications(
    query: str, candidates: Sequence[ApplicationCandidate], limit: int = 5,
) -> tuple[ApplicationMatch, ...]:
    if limit < 1:
        raise ValueError("Application candidate limit must be positive.")
    matches = [ApplicationMatch(candidate, _score(query, candidate)) for candidate in candidates]
    plausible = [match for match in matches if match.score >= 70 and match.candidate.launch_policy != "deny"]
    plausible.sort(key=lambda match: (-match.score, normalize_app_name(match.candidate.display_name), match.candidate.id))
    return tuple(plausible[:limit])


class MemoryApplicationCatalog:
    """Deterministic catalog used by tests and embedders with trusted callbacks."""

    def __init__(
        self, candidates: Sequence[ApplicationCandidate],
        launchers: dict[str, Callable[[], None]] | None = None,
    ) -> None:
        self._candidates = tuple(candidates)
        self._by_id = {candidate.id: candidate for candidate in self._candidates}
        self._launchers = launchers or {}

    def discover(self) -> tuple[ApplicationCandidate, ...]:
        return self._candidates

    def find(self, query: str, limit: int = 5) -> tuple[ApplicationMatch, ...]:
        return match_applications(query, self._candidates, limit)

    def resolve(self, app_id: str) -> ApplicationCandidate | None:
        return self._by_id.get(app_id)

    def launch(self, app_id: str) -> None:
        candidate = self.resolve(app_id)
        if candidate is None or candidate.launch_policy != "allow" or app_id not in self._launchers:
            raise ValueError("Application ID is unavailable or not approved for launch.")
        self._launchers[app_id]()

    def identify(self, process_name: str, package_family: str = "") -> str | None:
        process = normalize_app_name(process_name.removesuffix(".exe"))
        package = package_family.casefold()
        matches = [candidate.id for candidate in self._candidates if (
            package and candidate.package_family.casefold() == package
        ) or any(normalize_app_name(name.removesuffix(".exe")) == process for name in candidate.process_names)]
        return matches[0] if len(matches) == 1 else None
