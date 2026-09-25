# voice-jev

A Python 3.13 foundation for an open-source Windows 11 voice-controlled
computer-use agent. Inspired by [shhivv/third-hand](https://github.com/shhivv/third-hand),
with a new Windows architecture; no macOS source is ported.

**Generic installed-application discovery, UI observation, supervised typed
execution, hybrid UIA/window-capture observation, Jev decisions, and a bounded
request loop and observation-only OpenRouter, Gemini, DeepSeek, and OpenAI visual adapters are implemented.**
One-shot CLI push-to-talk capture and speech transcription are also available.
No global shortcut, wake word, continuous listening, TTS, OCR pipeline, real
visual coordinate execution, or unbounded autonomy is included.
Automated tests mock Windows and never control the desktop.

## Setup and verification

Install Python 3.13 on Windows 11, then run in PowerShell from the project root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe main.py observe
```

The CLI automatically loads `.env` from the project directory. Copy
`.env.example` to `.env` and add local credentials there, or set them in the
PowerShell session. Existing process environment variables take precedence over
`.env`; the CLI does not print API keys or `.env` contents. `.env` is ignored by
Git.

No virtual environment activation is required. `decide` and `run-agent` require
an API key; observation, manual actions, and tests do not. pywinauto is a
Windows-only dependency, loaded only for real desktop operations.
Dependency ranges are intentional; no lockfile is supplied.

## Observe a window

Run this, then switch to your target application during the five-second delay:

```powershell
.\.venv\Scripts\python.exe main.py observe --delay 5
```

The command snapshots the foreground window handle without activating anything,
then uses `Desktop(backend="uia")` and lazy `UIAElementInfo.iter_children()`
to inspect it. Without a delay, launching from a terminal normally observes that
terminal. No screenshots, OCR, input actions, or network requests are used.

Output is indented JSON containing `window_title`, `process_id`, `control_type`,
best-effort executable and packaged-app identity, an opaque catalog application
ID when one can be resolved, and an `elements` list. Each element has an
observation-local `id` (`c1`, `c2`,
...), name, control type, automation ID, rectangle, enabled state, and visibility.
For the focused Edit/Document only, `observed_text` may contain a bounded,
credential-redacted accessibility value and `observed_text_truncated` says whether
the limit was reached. TextPattern is read first with a bounded `GetText`; the
observer falls back to ValuePattern when TextPattern is unavailable.
The generic `Observation`/`UIElement` models use `Rect` for bounds. `app_name` is
the best-effort foreground executable basename. `application_id` is populated
only when the executable or package family uniquely matches the trusted local
catalog. Permission failures leave unavailable identity fields empty.
Missing text is an empty string; unavailable bounds, PID, or enabled state is null.
`ClickAction(target_id="c7")` is the action syntax; execution also requires the
exact observation object from the same `WindowsComputer` instance.

Default limits are configurable:

```powershell
.\.venv\Scripts\python.exe main.py observe --max-depth 6 --max-controls 100 --max-nodes 500 --max-text-length 200 --max-observed-text-length 500
```

- The foreground window is depth zero. `--max-depth 0` reads metadata only.
- Traversal is depth-first. Invisible branches are skipped. Unknown visibility
  prevents emitting that control but permits inspecting its children.
- Empty layout nodes are omitted while their children remain eligible. Unnamed
  interactive controls such as buttons and edit boxes are retained. Disabled
  controls remain useful context and are included.
- The node budget counts filtered controls too. Text fields are length-limited.
- Editable content is read only when UIA positively reports focused and
  non-password. Unknown/password fields are never queried. Known environment
  credentials and common token/credential forms are redacted before storage.
- `truncated` means a traversal/output/text boundary was reached; it is conservative
  and may be true even if the final visited node had no additional children.
- `inspection_errors` counts caught property/traversal failures. Partial results
  survive; an iterator failure abandons that branch's remaining siblings and
  resumes at its ancestor. Failure to obtain the window returns a JSON `error`
  and exit status 1. Partial observations return status 0; invalid CLI options use 2.

IDs remain fixed in the returned immutable snapshot and restart with each call.
`observation_id` identifies a snapshot in output, but is not an execution token.
The executor retains UIA references privately; serialized observations cannot
authorize execution in another process.

## Execute one action manually

Run from a normal interactive Windows desktop. The action commands print a typed
result as JSON with `success`, `action`, `message`, `error`,
`requires_confirmation`, and `completed`. Exit status is 0 for success and 1 for
a denied/failed action; invalid CLI arguments use 2. Text appears in result JSON,
so do not use sensitive text in debug commands or persist the output carelessly.

```powershell
.\.venv\Scripts\python.exe main.py list-apps
.\.venv\Scripts\python.exe main.py find-app "Spotify"
.\.venv\Scripts\python.exe main.py open-app "Notepad"
.\.venv\Scripts\python.exe main.py press-key tab --delay 5
.\.venv\Scripts\python.exe main.py press-key shift+tab --delay 5
.\.venv\Scripts\python.exe main.py type-text "Hello from voice-jev" --delay 5
.\.venv\Scripts\python.exe main.py inspect-and-click --delay 5
```

For key/text commands, switch to the intended window and focused control during
the delay. The CLI observes immediately before execution. `type-text` **replaces
the entire focused editable field's value**, rather than appending or typing at
the caret. It uses native UIA Value.SetValue, never a keyboard parser; braces,
modifiers, Unicode, and newlines remain literal text. It never sends Enter.
Password fields, read-only fields, and controls without ValuePattern are refused.
There is no keyboard fallback for text in this version.

`inspect-and-click` waits, observes, and prints the snapshot. Return to the
terminal and enter an ID from that output. Then switch back to the **same window**
during the second delay. The command executes against the original snapshot;
it never silently re-observes and remaps the ID. Blank input cancels. There is
intentionally no standalone `click c7` command.

### Binding and safety rules

- A snapshot belongs to one executor instance, expires after 120 seconds, and is
  invalidated by a newer observation or **any execution attempt**. Observe again
  between actions, including after focusing an editor. Use one instance on one
  thread; this is not a concurrent desktop controller.
- Before acting, the adapter rechecks foreground HWND, retained UIA runtime IDs,
  process ID, name/type/automation ID, visibility, enabled state, and ancestry.
  Typing and keys also require the same focused element captured at observation.
  Copied, foreign, expired, changed, unbound, or consumed targets fail closed.
- Clicking uses Invoke, SelectionItem, or Toggle as appropriate. Clicking an Edit
  or Document uses UIA SetFocus and verifies focus. Unsupported patterns fail;
  there is no coordinate-based fallback.
- Application launch accepts only an opaque ID resolved by the local catalog.
  The catalog discovers Start Menu `.lnk` files under the standard trusted Start
  Menu roots and packaged applications exposed by `shell:AppsFolder`. It invokes
  the retained shortcut or Shell object; Jev never receives a path, argument,
  command line, or native object. No disk-wide scan, PATH search, `cmd`,
  PowerShell, or `shell=True` execution is used.
- Command prompts, PowerShell/Terminal, registry and administration consoles,
  script hosts, and common executable interpreters are denied using display
  names and resolved executable identity. Unknown and confirmation-classified
  applications fail closed.
- Only `enter`, `escape`, `tab`, `shift+tab`, `ctrl+a`, and `ctrl+c` are accepted
  (key names are case-insensitive). Only fixed internal expressions reach
  pywinauto's keyboard backend.
- `BasicActionPolicy` enforces the baseline. An optional `ActionPolicy` may further
  deny or return `confirm`; confirmation returns `requires_confirmation: true`
  without performing the action. No consent-resume workflow is implemented yet.
- `AutonomousActionPolicy` requires confirmation for Enter and controls whose
  accessible labels indicate sending, submission, purchase, deletion, upload,
  posting, authentication, or password changes. Matching uses normalized whole
  words and short phrases in English and Spanish, including `delete`/`eliminar`,
  `send`/`enviar`, `buy`/`comprar`, `upload`/`subir`, and
  `publish`/`publicar`. With no confirmation provider, the loop stops. This is a
  compact deterministic vocabulary rather than a complete intent classifier.
- Action failures may follow partial effects. Actions are never retried and no
  fallback input is attempted. Inspect the UI before deciding what to do next.

Programmatic usage also requires explicit snapshot ownership:

```python
from computer.actions import ClickAction, FinishAction
from computer.windows_actions import WindowsComputer

computer = WindowsComputer()
observation = computer.observe()
# Choose an actual ID from observation.elements, in this process:
result = computer.execute(ClickAction(target_id="c7"), observation)
finished = computer.execute(FinishAction("Manual task complete"))
```

`FinishAction` returns `completed=True` without reading or controlling Windows.

### Discover and inspect applications

The same `WindowsApplicationCatalog` is used by diagnostics, Jev candidate
construction, safety validation, the executor, and foreground verification.
Candidate IDs are stable for the same local source reference but intentionally
opaque, for example `app_8a4e...`. Matching normalizes Unicode, punctuation,
case, and common request words, ranks exact and strong token matches first, and
returns at most five candidates by default. The budget is configurable through
the catalog and `JevDecisionMaker` constructors. An explicit application launch
request with no safe plausible match returns `needs_human` before the API call.

These read-only diagnostics do not launch anything:

```powershell
.\.venv\Scripts\python.exe main.py list-apps
.\.venv\Scripts\python.exe main.py find-app "Spotify"
.\.venv\Scripts\python.exe main.py find-app "Chrome"
.\.venv\Scripts\python.exe main.py find-app "WhatsApp"
.\.venv\Scripts\python.exe main.py inspect-app --delay 5
```

`list-apps` and `find-app` print only ID, display name, publisher, source,
policy, and optional match score. Private shortcut paths, executable paths,
arguments, AUMIDs, process mappings, and launch bindings are omitted.
`inspect-app` lets you switch to any application, then reports sanitized window
identity, counts by UIA control type, up to 40 named interactive controls, one
immediate parent label/type, and only presence/length metadata for focused
editable text. It never prints editable values or password controls.

## Hybrid visual observation

UIA remains the primary source. A deterministic local policy requests visual
supplementation only when UIA has no useful named controls, is sparse, lacks an
editor for a text-oriented goal, or is truncated with little useful content.
The policy uses control quality and request shape, never application names.

When vision is useful, `WindowsWindowCapture` verifies the default interactive
desktop and exact foreground handle, rejects minimized/invalid/secure windows,
clips the window to the virtual multi-monitor desktop, and captures only that
visible window region. Capture metadata records full window bounds, clipped
capture bounds, pixel dimensions, per-window DPI, and measured pixels-per-screen-
coordinate scale. Negative monitor coordinates and partially off-screen windows
are supported. Geometry with an invalid scale fails closed.

Known UIA password rectangles are painted black before a provider sees the
image. Provider text is bounded and passed through the existing credential
redactor. Requests containing detected credentials suppress capture. Screenshots
are closed immediately after provider processing and are never placed in an
`Observation`, Jev payload, debug JSON, or history. A remote provider would still
receive visible application pixels outside known password regions, so enabling
one has a broader privacy profile than UIA-only operation.

`VisualObserver` is the provider-neutral boundary. It receives ephemeral pixels,
sanitized window context, and the original request, then returns untrusted
`VisualCandidate` records. Local validation checks types, optional confidence,
bounds, dimensions, text limits, and element limits before assigning `v1`, `v2`,
and similar snapshot-local IDs. `FakeVisualObserver` provides deterministic test
coverage. Each remote adapter remains isolated under `computer/visual_providers/`.
Provider selection is explicit at runtime; a provider error never calls another
provider automatically.

The explicit free-model allowlist contains `google/gemma-4-26b-a4b-it:free` and
`inclusionai/ling-3.0-flash-vl:free`. Both catalog entries currently list image
input, text output, and zero input/output token prices. The code rejects
`openrouter/free`, text-only models, paid model IDs, and every model not present
in this allowlist. Gemma uses documented JSON mode through
`response_format: {"type":"json_object"}`. Ling does not support
`response_format`; it uses its documented `tools` and forced `tool_choice`
capabilities to return exactly one function call whose arguments must pass the
same strict JSON and bounding-box validation. Plain text, missing/multiple calls,
the wrong function, Markdown fences, and malformed arguments fail closed.
Both strategies use `provider.allow_fallbacks: false` and request providers that
deny data collection; if no matching endpoint exists, observation fails closed.

DeepSeek's current official API documents image input for `deepseek-flash`.
That adapter uses the OpenAI-compatible Chat Completions endpoint and documented
JSON mode. It is paid and reported as `very_low_cost`; no free DeepSeek API tier
is assumed. `deepseek-chat` and undocumented aliases are rejected locally.
Official references: OpenRouter [image inputs](https://openrouter.ai/docs/guides/overview/multimodal/image-understanding),
[Gemma free model card](https://openrouter.ai/google/gemma-4-26b-a4b-it%3Afree),
[Ling 3.0 Flash VL free model card](https://openrouter.ai/inclusionai/ling-3.0-flash-vl%3Afree),
[tool calling](https://openrouter.ai/docs/guides/features/tool-calling),
[structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs),
[free rate limits](https://openrouter.ai/docs/faq), and
[privacy policy](https://openrouter.ai/privacy/); DeepSeek [vision guide](https://api-docs.deepseek.com/guides/vision/),
[JSON Output](https://api-docs.deepseek.com/guides/json_mode/), and
[models and pricing](https://api-docs.deepseek.com/quick_start/pricing/).

The direct Google Gemini adapter defaults to stable `gemini-3.5-flash-lite`,
which Google documents as a low-latency multimodal model with image input, text
output, and structured outputs. It calls the official REST
`models/{model}:generateContent` method and requests native JSON Schema output
through `generationConfig.responseMimeType` and
`generationConfig.responseJsonSchema`. The API key is sent only in the
`x-goog-api-key` header. The metadata diagnostic calls `models.get`; capabilities
and account pricing that this endpoint does not expose remain `null` or
`account_tier_dependent`. Official references: [model card](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite),
[GenerateContent API](https://ai.google.dev/api/generate-content),
[structured outputs](https://ai.google.dev/gemini-api/docs/generate-content/structured-output),
and [Models API](https://ai.google.dev/api/models).

The retained OpenAI adapter uses the Responses API with `gpt-6-astra`, PNG bytes encoded
as a base64 data URL, `detail: high`, `store: false`, and strict Structured
Outputs through `text.format`. Official references: [GPT-6 Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create),
[image inputs](https://developers.openai.com/api/docs/guides/images-vision), and
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

The schema asks for integer boxes on a 0–1000 coordinate plane relative to the
submitted image. This is an application-defined output contract, not a native
coordinate system returned by the API. The adapter validates ordering and range,
then converts each edge into capture-relative pixels. Boxes smaller than three
pixels after conversion are rejected. Provider-specific coordinates never leave
the adapter.

The API does not provide a calibrated confidence for each grounded region, so
real candidates use `confidence: null`. No probability is invented and the
existing confidence threshold is unchanged. Real-provider visual actions are
also explicitly marked unauthorized, making this integration observation-only.

Equivalent UIA and visual elements are deduplicated only when normalized labels
and roles are compatible and their screen bounds strongly overlap. UIA wins.
Adjacent same-label controls remain distinct when their bounds do not overlap.

Jev sees descriptions such as:

```text
Click "Search" [button, visual, confidence unavailable] (v3).
```

That representation is available for diagnostics and for providers with locally
acceptable confidence. Current OpenAI elements have unknown per-element
confidence and are not released as executable Jev click options in this phase.

It never sees capture rectangles or constructs coordinates. A selected option
maps locally to `VisualClickAction(snapshot_id=..., target_id="v3")`. Before one
click, the executor checks the exact observation object, snapshot ID, target,
confidence, foreground UIA identity, unchanged window bounds, capture containment,
scale conversion, and deterministic rectangle center. Any mismatch returns a
stale/unsafe result and requires a fresh observation. Visual actions pass through
the same consequential-action policy as UIA actions.

Without provider configuration, the safe diagnostic retains its prior capture-
only behavior and reports `visual_provider_available: false`:

```powershell
.\.venv\Scripts\python.exe main.py inspect-hybrid --delay 5
```

It reports UIA quality, fallback reason, capture geometry, DPI/scale, masking
count, provider availability, and visual-control count without image bytes. To
inspect capture geometry manually, explicitly choose a new PNG path:

```powershell
New-Item -ItemType Directory -Force debug-screenshots
.\.venv\Scripts\python.exe main.py inspect-hybrid --delay 5 --save-debug-screenshot debug-screenshots\capture.png
```

To enable the recommended observation-only free provider for the current
PowerShell session, read the key without echoing it:

```powershell
$env:OPENROUTER_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'OpenRouter API key' -AsSecureString)).Password
$env:VISUAL_PROVIDER = 'openrouter'
$env:OPENROUTER_VISUAL_MODEL = 'inclusionai/ling-3.0-flash-vl:free'
$env:OPENROUTER_DATA_COLLECTION = 'deny'
$env:VISUAL_MAX_ELEMENTS = '40'
$env:VISUAL_TIMEOUT_SECONDS = '20'
$env:VISUAL_ACTIONS_ENABLED = 'false'
```

For a manual paid comparison, set `VISUAL_PROVIDER` to `deepseek` with
`DEEPSEEK_API_KEY` and `DEEPSEEK_VISUAL_MODEL=deepseek-flash`, or to `openai`
with `OPENAI_API_KEY` and the existing `VISUAL_MODEL`. Switching requires no
code change. Never configure several providers for one observation.

To benchmark direct Gemini for the current PowerShell session, read the key
without echoing it:

```powershell
$env:GEMINI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'Gemini API key' -AsSecureString)).Password
$env:VISUAL_PROVIDER = 'gemini'
$env:GEMINI_VISUAL_MODEL = 'gemini-3.5-flash-lite'
$env:VISUAL_MAX_ELEMENTS = '40'
$env:VISUAL_TIMEOUT_SECONDS = '20'
$env:VISUAL_ACTIONS_ENABLED = 'false'
# Optional one-shot fallback after a retryable Gemini provider failure.
$env:OPENAI_VISUAL_MODEL = 'gpt-5.6-luna'
```

When `VISUAL_PROVIDER=gemini` and `OPENAI_API_KEY` is present, directed visual
grounding tries Gemini once with a six-second maximum deadline. A timeout,
connection/rate-limit/server failure, or invalid structured response can trigger
one OpenAI Responses request using the same masked screenshot and grounding
objective. A valid empty Gemini result is accepted without fallback. The
fallback can incur OpenAI API charges; omit `OPENAI_API_KEY` to disable it.
Neither provider is allowed to click, type, or select an action.

Then run the diagnostic and manually switch to the target application:

```powershell
.\.venv\Scripts\python.exe main.py inspect-hybrid --delay 5
```

Generic grounding inventories up to `VISUAL_MAX_ELEMENTS` controls (40 by
default) using Gemini's existing five-field element schema and 8,000-token
output budget. Experimental directed grounding asks only for controls relevant
to one bounded semantic objective. Its compact remote schema contains label,
role, normalized box, and clickability; parent is reconstructed locally as an
empty value, while `vN` IDs and `confidence=None` remain local. It defaults to
five controls and an 800-token output budget:

```powershell
$env:VISUAL_DIRECTED_MAX_ELEMENTS = '5'
$env:GEMINI_VISUAL_DIRECTED_MAX_OUTPUT_TOKENS = '800'
.\.venv\Scripts\python.exe main.py inspect-hybrid-directed --objective "Find the control used to search for music" --delay 5
```

This command is observation-only and is not connected to `run-agent`. It reports
capture, request-build, provider, response-parse, and total visual-observation
timings plus requested and returned element counts. The objective, request
prompt, API key, and screenshot data are excluded from diagnostics. Visual
execution remains disabled. No latency improvement is claimed until the generic
and directed commands have been measured under comparable conditions.

To compare repeated grounding results without changing the desktop, capture the
foreground window once and run the four built-in objectives three times each:

```powershell
.\.venv\Scripts\python.exe main.py diagnose-visual-grounding --delay 5
```

Repeat `--objective "..."` to supply a custom objective set, and use `--runs`
to select 1–10 independent calls per objective. The command prints a prominent
no-actions warning, keeps the masked screenshot only in memory, and never
instantiates the Windows action executor. Each result includes bounded stage
counts, timing, usage, normalized candidate boxes, and a safe request
fingerprint. The fingerprint includes only request-shape metadata, encoded PNG
length, and a local SHA-256 digest; it excludes the objective text, prompt, API
key, image data, and provider response.

### Experimental Jev-controlled hybrid debug loop

`run-agent-hybrid-debug` is a separate experimental controller path. The stable
`run-agent` command and its Jev schema are unchanged. In the experimental path,
Jev still chooses every workflow step. Gemini remains an observation-only visual
grounder and receives only a short, bounded semantic target selected through a
typed `VisualGroundingNeed` option.

The loop observes UIA first. Safe local actions such as launching an allowlisted
application may execute under the existing policy. When UIA cannot expose the
next semantic target, Jev can request a directed visual observation. That creates
a fresh snapshot with locally assigned `vN` candidates, and Jev then chooses
among their labels and roles without receiving coordinates. If Jev selects a
`VisualClickAction`, the loop terminates successfully at the expected debug
boundary with `visual_action_blocked_for_debug`. It reports the selected ID,
label, role, snapshot, Jev confidence, and normal policy disposition, but it
never calls the mouse executor.

### Experimental single visual-click phase

`run-agent-hybrid-click-debug` is a separate opt-in milestone. It follows the
same UIA-first, directed-grounding, and visual-readiness path, but may attempt
one locally safety-validated `VisualClickAction`. The run consumes its click
budget before OS input, observes the foreground again after the attempt, records
bounded effect evidence, and terminates. It cannot type visually or perform a
second visual click. Targets with stale snapshot/window geometry, credential
context, unsupported roles, empty labels, or consequential label/request terms
are denied.

```powershell
.\.venv\Scripts\python.exe main.py run-agent-hybrid-click-debug "Open Spotify and focus the search field" --delay 5
```

The three modes remain distinct:

- `run-agent` keeps its established behavior.
- `run-agent-hybrid-debug` observes and decides but blocks every visual click.
- `run-agent-hybrid-click-debug` may issue one safe visual click, performs one
  fresh local observation, and stops with `visual_click_phase1_complete` when
  input and post-observation both succeed.

`run-agent-hybrid-type-debug` adds one further bounded milestone: after its one
safe visual click, it may issue one literal `TypeAction` and then stops. The
literal is selected deterministically from an existing `literal_texts` source
slice; for a request shaped as `play <title> by <artist>`, phase 2 uses the
bounded `<title>` slice. Jev and Gemini cannot invent the typed value.

Typing is released immediately when a fresh local observation exposes a
focused, enabled, non-password editor in the same foreground context. Otherwise
the controller makes one directed verification observation. Local policy then
requires the same HWND, PID, window/capture bounds, an editable/search-like
candidate, and at least 50% overlap relative to the smaller of the old and new
normalized target rectangles. Verification never causes a second click.

```powershell
.\.venv\Scripts\python.exe main.py run-agent-hybrid-type-debug "Open Spotify and play Californication by Red Hot Chili Peppers" --delay 5
```

`run-agent-hybrid-result-debug` is the separately confirmed Phase-3 experiment.
It reuses the bounded search-field click and literal typing path, captures a
masked post-Type baseline, submits that verified query with exactly one Enter,
then waits locally for a changed and stable submitted-results frame. It makes
one directed visual result observation and permits
one additional non-consequential result click only after exact local title
matching, optional compatible creator evidence, Jev confidence of at least
`JEV_MIN_CONFIDENCE`, and snapshot/foreground/geometry revalidation. Two
indistinguishable matches stop as ambiguous. Missing creator evidence is allowed
only when exactly one title match remains. The command always stops after one
fresh local post-result observation; it never submits a second Enter or
continues to playback controls.

```powershell
.\.venv\Scripts\python.exe main.py run-agent-hybrid-result-debug `
  "Open Spotify and play Californication by Red Hot Chili Peppers" `
  --delay 5 `
  --save-result-debug-screenshot .\spotify-phase3-result-grounding-3.png
```

### Experimental generic target activation

`run-agent-generic-debug` is a new, separately confirmed controller. It keeps
UIA first, adapts UIA and directed visual candidates into the same generic
resolver, and only asks Jev to choose from a small locally admissible frontier.
It can activate a target already visible or perform one bounded literal search;
Enter is sent only after the typed query is re-observed and no target resolved.
After one final target activation it takes one fresh local observation and
stops. The mode has fixed small budgets and blocks consequential targets.

```powershell
.\.venv\Scripts\python.exe main.py run-agent-generic-debug `
  "Open report.pdf" `
  --delay 5
```

This mode does not replace `run-agent` or migrate any hybrid phase. Its target
choice is limited to bounded semantic candidate evidence; Jev receives no
coordinates, window handles, or screenshot data.

Trusted app activation uses the same `OPEN_APP_ACTIVATION_TIMEOUT_SECONDS`
setting and polling interval as the hybrid debug mode (3 seconds by default,
200 ms polls). This generic debug mode first observes passively; after two
consecutive probes find the same unique, visible, catalog-matched window, it
makes at most one `SetForegroundWindow` call. A visible minimized target is
restored with `ShowWindowAsync(SW_RESTORE)` before that call. It then requires a fresh
foreground observation to succeed. Windows can reject foreground changes while
the user is working in another window, so this remains best effort and does not
simulate keyboard or mouse input. The ordinary `run-agent` and hybrid modes
remain unchanged. Diagnostics omit HWNDs, PIDs, window titles, package internals,
paths, and command lines.

See Microsoft's [`SetForegroundWindow`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setforegroundwindow),
[`ShowWindowAsync`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindowasync),
and [`IsWindow`](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-iswindow)
documentation for the API behavior and foreground restrictions.

The local result readiness wait defaults to seven seconds with 200 ms polling
and requires 1200 ms of quiet after the latest meaningful visual transition.
Configure it with `RESULT_READINESS_TIMEOUT_SECONDS`,
`RESULT_READINESS_POLL_INTERVAL_MS`, and `RESULT_SETTLE_QUIET_MS`. Polling
compares only bounded masked-frame statistics and does not call the visual
provider. If at least two of three richness ratios (PNG bytes per megapixel,
sampled unique colors, and luminance standard deviation) fall below 0.75 of the
pre-submit baseline, a stable candidate enters `awaiting_followup_transition`
for a separate 2500 ms grace period. Configure that period with
`RESULT_TRANSITION_GRACE_MS`; the ratio and required signal count can be tuned
with `RESULT_SIMPLIFICATION_RATIO_THRESHOLD` and
`RESULT_SIMPLIFICATION_MIN_SIGNALS`. A meaningful follow-up transition replaces
the candidate and restarts quiet settling. If the simplified candidate remains
stable through the grace period, Phase 3 fails closed before Gemini is called.
The current frame must also be informative and stable relative to a recent
capture. The accepted capture is reused for grounding without a recapture.

For an explicit Phase-3 semantic diagnostic run, add
`--save-result-debug-screenshot PATH`. It saves only the exact masked PNG used
for result grounding, refuses to overwrite an existing file, and prints its
SHA-256 and byte length for comparison with the request fingerprint. The result
trace includes the bounded grounding objective, provider/validation counts, and
redacted candidate label, parent, role, matching, geometry, and rejection data.

The click and type budgets are consumed before their respective OS input calls.
The Type implementation sends literal Unicode without clipboard or keyboard
macro syntax and never sends Enter. After typing, the controller obtains one
fresh local observation, records bounded match metadata without logging field
contents, and stops with `visual_type_phase2_complete`. It cannot select a
result or continue playback.

The strong-visual literal path uses Win32 `SendInput` with
`KEYEVENTF_UNICODE`. Its ctypes declarations include the complete native
`INPUT` union (`MOUSEINPUT`, `KEYBDINPUT`, and `HARDWAREINPUT`) and a
pointer-sized `ULONG_PTR`, so `sizeof(INPUT)` is 40 bytes on 64-bit Windows and
28 bytes on 32-bit Windows. Every UTF-16 code unit produces one Unicode keydown
and one Unicode keyup. Partial `SendInput` completion fails closed without
retrying the remaining events.

Before spending a Gemini call, the isolated input transport can be tested by
opening Notepad and running:

```powershell
.\.venv\Scripts\python.exe main.py debug-type-literal "Hello" --delay 5
```

The command requires confirmation and performs exactly one literal input
operation against the manually focused foreground window. It uses no Jev,
Gemini, clipboard, shell, subprocess, hotkey syntax, or continuation. Output
contains only bounded event counts, structure size, foreground identity,
failure stage, and the immediate Win32 error code when applicable.

Immediately before that directed provider call, hybrid debug evaluates visual
readiness locally. A frame is considered strongly suspicious only when at least
three of four signals agree: no more than 64 sampled colors, luminance standard
deviation no greater than 3, no more than 20,000 encoded PNG bytes per
megapixel, and an RGB channel range no greater than 48. Mean luminance is not a
rejection signal, so dark interfaces with text, cards, or other spatial
structure pass. Suspicious frames are recaptured every 200 ms for at most three
seconds by default. Each recapture requires the same fresh foreground HWND and
PID. The accepted capture is passed directly to the provider without another
capture. Configure the bounds with `VISUAL_READINESS_TIMEOUT_SECONDS` and
`VISUAL_READINESS_POLL_INTERVAL_MS`.

For capture debugging, the hybrid command can explicitly save the first exact
masked PNG supplied to the visual provider:

```powershell
.\.venv\Scripts\python.exe main.py run-agent-hybrid-debug "Open Spotify and play Californication by Red Hot Chili Peppers" --delay 5 --save-visual-debug-screenshot .\spotify-hybrid-masked.png
```

The path must be a new `.png` in an existing directory. This option is off by
default and prints a warning because the masked image can still contain visible
private UI. The saved bytes are checked against the request fingerprint's byte
length and SHA-256. Capture diagnostics report the fresh foreground HWND/PID,
window and virtual-screen geometry, ImageGrab screen-region backend, masking
count and area, and bounded pixel statistics. They do not perform OCR or expose
masked text. Visual execution remains disabled.

Application launch options use the catalog's trusted `application_id`. In the
hybrid path, a requested application is not offered again while a fresh
foreground observation identifies that exact application ID as active. Window
title similarity is not accepted as identity. If a later fresh observation
shows a different application, the trusted launch option may be offered again.
After a successful launch, the controller polls local UIA observations until the
requested ID becomes foreground or `OPEN_APP_ACTIVATION_TIMEOUT_SECONDS`
expires (default: `3`). This wait calls neither Jev nor the visual provider and
does not consume agent steps. A timeout preserves the actual latest foreground
state instead of assuming activation succeeded.

The trace records per-step observation kind, useful UIA count, Jev decision and
latency, grounding request length, provider latency, candidate count, execution
status, terminal reason, step latency, and total run latency. It excludes image
bytes, Base64, keys, authorization data, passwords, coordinates, and provider
reasoning. Repeating the same grounding need against the unchanged grounded
snapshot stops as non-progress. Executed actions and changed objectives require
fresh observations and invalidate the run-local grounding reuse.

Each decision trace also reports `offered_option_types`, the selected typed
option key, bounded grounding objectives/reasons/candidate limits, and semantic
application IDs/names. It never reports executable paths. This makes a
low-confidence stop distinguishable from a missing grounding option without
exposing the Jev prompt or provider reasoning.

Malformed TypeSafe results remain fail-closed. Hybrid trace diagnostics expose
only bounded HTTP/category metadata, response field names and primitive types,
opaque returned option IDs, and retry outcome categories. They never include a
prompt, response body, answer text, credential, or provider secret. Grounding
choices use deterministic opaque IDs such as `grounding_1` and `grounding_2`.

The experimental path also records a minimal `task_progress` value with
`query_required`, `query_entered_or_submitted`, and
`result_grounding_eligible`. For staged search/find/play requests, result
grounding is withheld until a successful validated literal type action is
followed by a fresh observation. A proposed or failed type, a requested visual
observation, a returned visual candidate, or a blocked visual click does not
advance progress. Jev still selects from its ordinary typed Choice and the
configured confidence threshold still applies.

When TypeSafe returns a valid Choice distribution, explicit hybrid traces add a
bounded `choice_probabilities` list containing only locally offered opaque IDs
and numeric probabilities. Unknown IDs invalidate the response and are never
reported as valid options.

TypeSafe documents Choice `probabilities` as the distribution over every
offered option and `choice` as its highest-probability option. Its separate
`confidence` is a statistic derived from the shape of that distribution; it is
not required to equal the selected option's probability. Hybrid debug therefore
reports `selected_option_probability` only as diagnostics and continues to gate
actions on `confidence`.

Retries are recorded as separate bounded `decision_attempts`. Each attempt
contains its result/category, safe HTTP metadata, response field/type shape,
offered or `[unoffered]` option marker, numeric confidence, selected probability,
probability count, and probability sum where parsing reached those fields. Final
response probabilities are never attributed to an earlier failed attempt.

Hybrid Choice construction distinguishes executor capabilities from actions
applicable to the current observation. UIA clicks require a visible, enabled,
supported control plus bounded semantic relevance from its name, role, parent,
automation ID, focus, and request terms. Type and editing keys require a focused
non-password editor and applicable literal/text evidence. Enter requires a
focused submit-like control, Escape requires dialog/popup evidence, and Tab or
Shift+Tab require an incomplete observation plus a focused, relevant recovery
path. Stop, trusted OpenApp handling, task-progress grounding, Finish, and
snapshot-local visual candidates keep their existing semantics. This filtering
applies only to `run-agent-hybrid-debug`; normal `run-agent` is unchanged.

The bounded `option_filter_summary` trace reports candidate and offered counts
for UIA clicks, typing, and key capabilities, plus aggregate reason codes such
as `insufficient_semantic_relevance`, `no_editable_focus`, `no_submit_context`,
`no_modal_context`, and `keyboard_navigation_not_justified`. It contains no UI
labels, coordinates, prompts, or secrets.

Hybrid decisions now carry an explicit typed effect: `observe`, `act`, or
`terminal`. Every Windows-mutating action remains `act` and must meet
`JEV_MIN_CONFIDENCE` before normal safety validation. A locally constructed
`VisualGroundingNeed` is `observe`: it may proceed below that action threshold
only when the current snapshot, foreground identity, fallback need, objective,
candidate bound, credential context, and local offered-choice identity pass the
bounded observation policy. Stop and Finish are terminal decisions.

An allowed observation never enters `computer.execute`. It captures only through
the existing foreground-window capture path, retains password-region masking,
provider timeout and response bounds, and returns untrusted snapshot-local `vN`
candidates to Jev. Directed snapshots cannot recursively request more grounding,
and the same snapshot/objective cannot invoke the provider twice. A later visual
selection is still an `act` decision and remains blocked by both the `0.80`
action gate and the experimental no-visual-execution boundary.

Hybrid traces expose `decision_effect`, `release_policy`, and a bounded
`observation_policy` result containing only eligibility, a reason code, and
whether confidence was required.

Directed visual observations expose bounded `visual_pipeline` stage counts for
provider output, parsing, validation, deduplication, observation handoff, and
Jev visual options. `visual_grounding_status` distinguishes candidates, a valid
empty result, provider or parse failure, validation or deduplication loss, and a
handoff inconsistency. Aggregate rejection reason counts never include raw model
responses, image data, prompts, coordinates, or rejected labels. A valid empty
directed result terminates as `visual_grounding_empty` after one provider call;
the controller does not repeat or broaden the request automatically.

With Gemini configured as above, run the first Spotify integration measurement:

```powershell
.\.venv\Scripts\python.exe main.py run-agent-hybrid-debug `
  "Open Spotify and play Californication by Red Hot Chili Peppers"
```

The command prints `EXPERIMENTAL HYBRID DEBUG` and `VISUAL EXECUTION: DISABLED`
before asking whether safe local/UIA actions may run. A stop at a semantic visual
target is the expected outcome in this phase. Directed grounding measured about
2–3 seconds for the tested Spotify search-field and result cases; that local
measurement is not a universal latency claim. The separate one-shot voice CLI
does not add voice logic to this hybrid path. TTS, speculative execution, and
autonomous visual clicking are not part of it.

To inspect directed boxes explicitly (the file can contain private visible
content and is never saved by default):

```powershell
New-Item -ItemType Directory -Force debug-screenshots
.\.venv\Scripts\python.exe main.py inspect-hybrid-directed --objective "Find the control used to search for music" --delay 5 --save-debug-overlay debug-screenshots\spotify-directed.png
```

Before uploading any screenshot, OpenRouter or Gemini configuration and metadata
can be checked without requesting a model generation:

```powershell
.\.venv\Scripts\python.exe main.py check-visual-provider
```

For OpenRouter this calls the documented authenticated `GET /api/v1/models/user`
endpoint and `GET /api/v1/models/{author}/{slug}/endpoints`. It reports model-level
capabilities separately from each endpoint's public name, status, pricing class,
supported parameters, and deterministic compatibility with the exact parameters
the adapter sends. It does not capture the desktop, upload pixels, include a
prompt, or invoke Chat Completions. Endpoint input modalities and data-collection
policy are reported as unknown because the current endpoint schema does not expose
them. `OPENROUTER_DATA_COLLECTION` defaults to `deny`; in
that mode each generation includes OpenRouter's documented
`provider.data_collection: "deny"` routing filter. OpenRouter metadata cannot
prove that this filter has a matching endpoint, so endpoint eligibility is
reported as unverified and the observation fails closed if none is available.

To explicitly permit endpoints regardless of their data-collection policy for
the current PowerShell session:

```powershell
$env:OPENROUTER_DATA_COLLECTION = 'allow'
```

In `allow` mode the request omits the per-request `data_collection` filter, which
is OpenRouter's documented default routing behavior. The underlying provider may
process or retain screenshot data under its own policy. Restore the restrictive
default with `$env:OPENROUTER_DATA_COLLECTION = 'deny'`. This setting changes only
OpenRouter routing: capture masking and redaction remain active, model fallback
remains disabled, the free-model allowlist still applies, and visual execution
remains disabled.

It reports sanitized visual IDs, Unicode labels, roles, nullable confidence,
clickability, parent, capture-relative rectangles, latency, pricing class, and basic token usage.
It never prints the API key, image data URL, raw request, or raw response.
Provider failures are represented by a bounded object containing a safe category,
optional HTTP status, sanitized provider code/type, and a fixed short message.
OpenRouter categories distinguish authentication, permission, payment, rate limit,
missing model, no eligible endpoint, provider availability, request/response-format
errors, timeout, network, server, malformed-response, and unknown API failures.
Provider response bodies and provider-supplied free-form messages are never printed.
Remote visual HTTP operations have a 20-second hard wall-clock deadline by
default. The deadline covers connection setup, TLS, request transmission,
response headers, and the complete bounded response body, including chunked
responses. `VISUAL_TIMEOUT_SECONDS` accepts values from 1 through 120. Gemini's
interactive request is capped at six seconds (or the shorter configured shared
timeout), and this adapter uses one HTTP attempt with no SDK retry/backoff. A
timeout preserves the UIA observation and appears as a structured provider error.

Remote screenshots can contain private visible content. OpenRouter is an
intermediary and sends image input to an underlying model provider; retention
and training policies can therefore depend on that provider. DeepSeek's public
privacy policy says supplied photos/files may be collected and data may be
processed and stored in the People's Republic of China. Password-region masking
and request redaction reduce exposure but cannot make a desktop screenshot
anonymous. Review the provider's current policy before using sensitive windows.

For an explicit grounding overlay:

```powershell
New-Item -ItemType Directory -Force debug-screenshots
.\.venv\Scripts\python.exe main.py inspect-hybrid --delay 5 --save-debug-overlay debug-screenshots\spotify-overlay.png
```

Screenshot and overlay files may contain private visible content. Saving is
disabled by default and both commands refuse to overwrite an existing file.

### Unsaved Notepad smoke test

1. Run `open-app "Notepad"` above and manually select a new empty document.
2. Run `inspect-and-click --delay 5`, switch to Notepad, then choose its Edit or
   Document control ID in the terminal. Switch back during the second delay.
3. Run `type-text "Hello from voice-jev" --delay 5`, then switch to the same
   Notepad editor. Confirm the text visually. **Do not save the document.**

If that Notepad version lacks UIA ValuePattern, text entry returns a failure
rather than emulating keys. In the first user-run autonomous tests, opening
Notepad and replacing its document value with `Hello from Jev` both succeeded.
The next decision lacked strong observable postcondition evidence: one response
failed schema validation, and a later Finish choice had 0.78 confidence and was
correctly blocked by the unchanged 0.80 threshold. Those results motivated the
postcondition and bounded-retry work below; they do not establish broad UIA or
model reliability.

## Ask Jev for one decision (no execution)

1. Sign in to the [official TypeSafe console](https://console.typesafe.ai/) and
   create an API key. Access may depend on your account invitation; use the
   [TypeSafe website](https://typesafe.ai/) if you need to request access.
2. Set environment variables in the same PowerShell session. This prompt avoids
   placing the key itself in shell history:

```powershell
$env:TYPESAFE_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'TypeSafe API key' -AsSecureString)).Password
$env:JEV_MIN_CONFIDENCE = '0.8'
$env:JEV_MODEL = 'jev-latest'
.\.venv\Scripts\python.exe main.py decide "click the Search button" --delay 5
.\.venv\Scripts\python.exe main.py decide "click the text editor" --delay 5 --debug-context
```

Switch to the target application during the delay. `decide` reads the foreground
window and sends compact context to TypeSafe, then prints one `DecisionResult`.
**It never clicks, types, sends keys, opens apps, or calls the executor.** A ready
result is a proposal, not execution or proof of task completion. No
`decide-and-act` command is provided in this step.

`--debug-context` prints the exact sanitized HTTP request body (state, instructions,
and semantic Choice descriptions) plus local selection counts to **stderr** before
the API call. The decision result stays on stdout. No authentication header, API
key, raw observation, UIA object, or geometry is included. Diagnostics also appear
if the API call fails; invalid/sensitive input is rejected before a debug payload
is produced. Delay prompts also use stderr, so redirected diagnostics are a log,
not necessarily one JSON document. Debug output contains ordinary window labels
and your request: credential filtering does not make it safe to publish blindly.

The environment template lists the supported variables. A missing key fails
before desktop inspection or networking.
Use `--max-controls 40` if a large UI exceeds the decision payload budget.

### Verified TypeSafe integration

The implementation uses Python's standard-library HTTPS client to send
`POST https://api.typesafe.ai/v1/systemone` with Bearer authentication and
`model`, `state`, and `questions`. This is the official typed decision API,
not a chat endpoint and not a request for generated JSON. No extra SDK dependency
is needed. There is a 15-second socket timeout, a bounded response size, no
redirects, and no automatic retries. HTTP/network failures return no action.

Official sources checked for this implementation:

- [HTTP API reference](https://docs.typesafe.ai/api)
- [Choice primitive](https://docs.typesafe.ai/primitives/choice)
- [Confidence semantics](https://docs.typesafe.ai/confidence)
- [Quick start and API key setup](https://docs.typesafe.ai/introduction/quickstart)
- [Official Python SDK alternative](https://docs.typesafe.ai/sdk/python)

Third Hand's [Jev client](https://github.com/shhivv/third-hand/blob/master/Sources/ThirdHand/JevClient.swift)
and [literal text planning](https://github.com/shhivv/third-hand/blob/master/Sources/ThirdHand/TextEntryPlan.swift)
informed the separation of state, offered targets, and local decoding. No macOS
implementation was ported. This version uses one complete-action Choice rather
than multiple operation/target questions.

### Decision schema and text constraints

`questions.next_action` is a single `choice` question. Its criteria are constructed
locally for each snapshot: `click_c1`, `open_app_<opaque-id>`, `key_tab`, `type_1`, `finish`,
and similar options. Each option has a semantic string description and maps to an
existing typed action object. `stop` maps
to human intervention rather than an action. Jev cannot provide an arbitrary
target, application, key expression, action payload, or text string.

### Representation revision: labels, focus, and completion

The previous `click_c1` criterion contained only
`{"kind":"click","target_id":"c1"}`. Jev did receive the name, Document role,
and focus flag, but only in a separate `state.controls` row. It had to join that
row to the opaque action. Controls were capped in observation order with no
presentation filtering or focus preference. Finish's description asserted task
completion without defining the distinction between an existing focused state
and a newly requested click. These are confirmed representation weaknesses and
plausible contributors to the reported finish/stop choices, not proof of the
model's internal reasoning.

The [official Choice API](https://docs.typesafe.ai/primitives/choice) accepts
semantic descriptions as criteria values. Now, for a generic observed Document:

```json
{
  "click_c1": "Click \"Editor de texto\" [Document, focused, enabled] (c1). Focus this document/text-editing area.",
  "click_c10": "Click \"Archivo\" [MenuItem, focus unknown, enabled] (c10). Activate this menu item.",
  "click_c13": "Click \"Editar\" [MenuItem, focus unknown, enabled] (c13). Activate this menu item."
}
```

These descriptions are data for Jev, not response parsers. Returning `click_c1`
still maps directly to the locally retained `ClickAction(target_id="c1")`; no
name, ID, or command is parsed from model-generated text. All action types have
effect descriptions. Document/Edit use general role descriptions across apps
and languages; no Notepad, localized-label, or request-specific scoring exists.

The compact state prominently includes `focused_controls` as well as window/app
labels and the selected controls. Focus means the current input target, not
proof that a requested action has happened. Finish explicitly means **all work
has already been completed**, with observed outcome or relevant successful
history as evidence. Stop explicitly means **no safe supported offered action
can progress the unfinished task**. Neither is a generic alternative to an
available requested action. The confidence threshold remains unchanged.

Fresh observations add explicit `completion_evidence` without declaring the
task complete. After a successful TypeAction, evidence is true only when the
new focused non-password editor exposes an untruncated value exactly matching
the requested literal. History alone leaves that match false. After a successful
OpenAppAction, evidence records whether the fresh foreground executable or
package identity resolves to the selected opaque catalog ID. No title heuristic
or application-specific branch is used. Jev must still explicitly choose Finish.
Raw editable content is not sent in control context; Jev sees presence, bounded
length, truncation, and requested-literal-match booleans.

Decision-only selection preserves the raw observation and original IDs:

- Apply the existing visibility/privacy filters, then rank focused controls first,
  enabled interactive roles second, and disabled/unknown-state interactive roles
  next. Ties preserve observation order. Apply the 80-control budget after ranking.
- Preserve Button, MenuItem, Document, Edit, Hyperlink, ListItem, TabItem, CheckBox,
  RadioButton, ComboBox and other known interactive roles. Selection does not
  expand the executor's supported action types.
- Omit unfocused Text consisting only of private-use icon glyphs; preserve mixed
  readable labels and actionable controls even if their label is an icon.
- Omit same-label Text only when its bounds are contained in a corresponding
  interactive control's bounds. Bounds are used locally, never sent to Jev. With
  no containment evidence, retain the duplicate Text but rank it last. Two
  actionable controls with identical labels are never merged.
- Omit empty unidentifiable Pane/Group/Window containers. Preserve named unknown
  controls, identifiable containers, focused controls, and unique status text.
- Intentional presentation removal alone no longer marks the observation
  incomplete. Privacy exclusions, budget omissions, or raw traversal failures/
  truncation still do, since they may hide relevant information.

Each control may include one immediate parent name/type. This gives context such
as `Daft Punk [ListItem]` under `Search results [List]` without serializing an
ancestry chain or the accessibility tree. Debug selection statistics report observed/eligible/selected control counts,
presentation/privacy/budget omissions, eligible/selected click-option counts, and
the total offered options. In the offline regression fixture, **8 controls become
4**, while **3 click options remain 3** and total options are **11** when no
installed application matches. Presentation
Text was never separately clickable, so it is correct that option count does not
drop in this fixture. This is a synthetic representation test, not a live API
quality result. The complete local suite includes model, representation,
Windows adapter, Jev, CLI, and loop tests.

The implementation process has no `TYPESAFE_API_KEY`, so it did not automatically
repeat the user-run Notepad tests. Use the commands below from the configured
interactive Windows session.
Remaining limits include provider-dependent UIA labels, only one bounded parent
level, lack of interaction-pattern metadata in the decision payload, and no measured accuracy improvement until a live
evaluation is performed. Ranking cannot recover controls absent from the raw
bounded observation, and semantic descriptions cannot guarantee a correct decision.

Click options include only currently offered enabled, visible controls passing
the baseline policy. Application candidates come from the trusted catalog and
key candidates reuse the executor allowlist.
Typing requires a positively observed focused, enabled, non-password Edit or
Document. The text itself comes only from an exact substring of the current
request, extracted locally from simple English `write`, `type`, `enter`, or
`search for` phrases:

- `write hello world` offers `hello world`.
- `search for Adele` offers `Adele`.
- `write "a short story"` offers those exact supplied words.
- `write me a 500 word essay about Rome` offers no text action.

Quoted content is recommended for ambiguity, punctuation, or longer prose.
This parser is intentionally conservative: it is not multilingual or a general
natural-language text extractor. It never synthesizes an essay, translates text,
or recycles text from UI labels/history. Future free-form generation belongs in
a separate component. Existing TypeAction semantics still replace the whole
editable value, not insertion at the caret.

### Confidence, validation, and data boundaries

Jev's returned `confidence` must be finite and at least `JEV_MIN_CONFIDENCE`
(default `0.8`, configurable in `(0, 1]`). The winning option's probability is
**not** substituted for confidence, and 0.8 confidence is not a promise of 80%
correctness. This is an initial threshold, not an empirically tuned safety claim.
Pin an available version with `JEV_MODEL` when comparing behavior over time.

The response must contain a valid Choice, one of the exact offered options,
and a complete finite probability distribution over those options. Local mapping
and baseline safety validation follow. Only a `ready` result contains an action.
Low confidence and `stop` yield `needs_human` with `action: null`; malformed
responses and network failures yield `error` with no action. Output includes
confidence, selected option, probabilities, model, and local snapshot ID; no
natural-language reasoning or raw response body is used. CLI exit status is 0
only for `ready`, otherwise 1.

During `run-agent`, `api_error` and `invalid_response` receive at most one retry
for that decision step. The retry uses the identical observation object and
identical recent history, and no action occurs between attempts. A second failure
stops closed. Low confidence, `stop`, invalid local input, safety rejection,
execution failure, and observation failure are never retried. Debug events show
safe reason codes such as `missing_model_metadata` or
`probability_option_mismatch`; response bodies and exception text remain hidden.

State contains the original request, bounded window/app labels, prominent focused
controls, up to 80 ranked visible UIA controls and validated visual controls
(semantic IDs, labels, roles, confidence, and optional parent only), editable
postcondition flags, completion evidence, and the last five action results. The
request is limited to 4000 characters, literal text to 500, and the complete
payload to 24 KB. Oversized contexts fail instead of silently shortening the
user's request. History uses typed actions and safe error codes, not raw error
messages. Historical control IDs are explicitly marked as belonging to older
snapshots. No PID, rectangle, automation ID, native object, screenshot, or other
window is sent.

Password-marked controls, and editable controls whose password status is unknown,
are omitted. Known credential environment values and
common credential patterns are redacted from labels/history; detected credentials
in the request cause local rejection. Credential filtering is best-effort, not
a general secret detector: do not use `decide` on sensitive windows or requests.
The API key is sent only as the authentication header, never state or logs. HTTP
error bodies and exception details are not printed. Labels remain untrusted data
and the model is instructed not to follow instructions embedded in them.

The decision maker accepts recent `ActionResult` values, and the generic loop
coordinates one bounded request:

```python
from decision.jev import JevDecisionMaker

decision_maker = JevDecisionMaker.from_environment()
decision = decision_maker.decide(user_request, observation, history=recent_results)
# The Agent performs the safety gate and exact-snapshot execution around this.
```

For a supervised run, use `python main.py run-agent "click the Search button"`.
The command asks for explicit confirmation before executing. Add `--dry-run` to
observe and obtain one Jev proposal without executing it.

The next bounded Notepad checks are:

```powershell
python main.py run-agent "Open Notepad" --max-steps 8 --debug
python main.py run-agent "Open Notepad and write 'Hello from Jev'" --max-steps 8 --debug
```

This implementation session did not rerun the full agent because its process had
no `TYPESAFE_API_KEY`; retry and evidence tests use deterministic fakes. The two
real user-run Notepad results described above are the current live evidence.

### UIA limitations

UIA data depends on each application's accessibility provider. Custom-rendered,
virtualized, elevated, or protected content may be missing or inaccessible.
Visibility uses UIA's offscreen state, not pixel-level occlusion detection.
Observation reads UIA names and, for the focused known non-password editor, up
to 500 characters through TextPattern or ValuePattern. ValuePattern itself has
no bounded read API, so the provider may materialize a larger BSTR before the
observer immediately truncates it; this is a UIA limitation. Some editors expose
neither pattern. Text execution requires the editable Value pattern. Semantic-only
clicks cannot operate every custom control.
The desktop can change during traversal; this is not an atomic snapshot.
Individual provider calls can block: depth/node limits bound traversal work but
do not impose a hard timeout on a hung COM call. pywinauto can also internally
turn some provider failures into empty properties, so the error count is best-effort.
Validation and execution are not atomic Windows operations. Another application
or person can change focus after the final check, especially during keyboard
emulation. Keep the desktop steady during a debug action. UIA runtime IDs are
provider-issued, temporary identities, not a durable security boundary.

See the [pywinauto UIA element API](https://pywinauto.readthedocs.io/en/latest/code/pywinauto.uia_element_info.html).

### Verification performed

The Windows smoke test on Python 3.13.15 observed the foreground Stremio window:
title, process ID, window type, and four controls (Pane/Text) with default limits,
zero caught inspection errors, and `truncated: true`. This verifies real UIA reads,
not comprehensive interactive-control coverage. The sandbox exposed no foreground
window and produced a structured error; the regular desktop session succeeded.
No application was launched or manipulated for this test.

## Architecture

| Module | Responsibility |
| --- | --- |
| `voice/models.py` | In-memory audio, transcript, and safe voice diagnostic models |
| `voice/interfaces.py` | Provider-independent microphone and speech-to-text contracts |
| `voice/audio.py` | Bounded Windows Enter/Enter push-to-talk WAV capture |
| `voice/openai_stt.py` | OpenAI `gpt-transcribe` adapter with bounded retry and sanitized errors |
| `voice/service.py` | One-shot orchestration and whitespace-only transcript normalization |
| `computer/actions.py` | Frozen typed action requests and `Action` union |
| `computer/applications.py` | Platform-neutral catalog, candidate, matching, and test catalog contracts |
| `computer/models.py` | Platform-neutral UI elements and observations |
| `computer/visual.py` | Provider contract, fake provider, validation, fallback, geometry, and UIA deduplication |
| `computer/visual_providers/common.py` | Shared prompt, PNG encoding, strict JSON validation, redaction, and normalized-box conversion |
| `computer/visual_providers/openrouter.py` | Pinned free multimodal Chat Completions/JSON-mode adapter |
| `computer/visual_providers/gemini.py` | Direct Gemini GenerateContent/native structured-output adapter |
| `computer/visual_providers/deepseek.py` | DeepSeek Flash multimodal Chat Completions/JSON-mode adapter |
| `computer/visual_providers/openai.py` | Retained observation-only OpenAI Responses API adapter |
| `computer/interfaces.py` | Generic Observer and Computer contracts |
| `computer/windows.py` | Bounded pywinauto/UIA Windows observer |
| `computer/windows_capture.py` | Ephemeral foreground-window capture, DPI metadata, masking, and explicit debug export |
| `computer/windows_apps.py` | Trusted Start Menu/AppsFolder discovery, private launch bindings, and foreground identity |
| `computer/windows_actions.py` | Bound sessions, semantic execution, Windows input/launch isolation |
| `computer/results.py` | Typed platform-neutral ActionResult |
| `decision/interfaces.py` | Decision contract with recent action history |
| `decision/client.py` | TypeSafe HTTP transport and environment configuration |
| `decision/context.py` | Control selection, compact state, literal extraction, and fresh-state completion evidence |
| `decision/options.py` | Semantic descriptions for locally bound Choice options |
| `decision/jev.py` | Local action space, Choice decoding, confidence gate |
| `decision/models.py` | Typed DecisionResult with safe error diagnostics |
| `safety/interfaces.py` | Allow/deny/confirm decisions and consent boundary |
| `safety/policy.py` | Mandatory catalog/key/target checks and generic consequential-action confirmation boundary |
| `agent/loop.py` | Bounded coordinator, postcondition debug summaries, and one safe decision retry |
| `main.py` | Observe, manual action commands, non-acting decide, and supervised `run-agent` |

Planned flow:

```text
CLI Enter-to-start/Enter-to-stop -> recording -> transcription -> visible transcript
  -> application discovery -> foreground window
  -> UIA + conditional capture/visual provider -> unified observation
  -> decide -> validate / confirm -> act -> fresh observation -> ...
                        -> FinishAction -> stop
```

The bounded coordinator wires one request through `observe -> decide -> safety
-> execute -> settle -> observe`. It passes the exact observation object used
for the decision into the executor, so snapshot-local control IDs cannot be
reused against a later window. Each iteration releases at most one action. A
FinishAction ends successfully without a Windows call; low confidence, stop or
error decisions, policy denials, failed execution, stale observations,
repetition, interruption, and the step limit terminate the run.

Run it manually with `python main.py run-agent "..."`. Real runs print a warning
and require `y`/`yes`; `--dry-run` observes and asks Jev for one proposal without
executing it. `--max-steps` overrides `AGENT_MAX_STEPS` (default 8), and
`--debug` prints step/status/count metadata, focused-editor value presence/length,
completion booleans, and safe retry reason codes to stderr. No request text,
control labels, observed values, response bodies, or typed literals are included.

The agent accepts a text request so voice capture remains independent of the
computer loop. Contracts are synchronous for now. The coordinator handles
cancellation, errors, denied actions, and a bounded step count. The first voice
adapter is available as a CLI input path; global-shortcut integration remains
future work.

## One-shot voice input

Install the normal project dependencies, set `OPENAI_API_KEY` in `.env`, then
run `python main.py voice-transcribe` for one push-to-talk recording. Press
Enter to start recording and Enter again to stop it (maximum 15 seconds by
default). The command displays the transcript and diagnostics, and never
creates a computer or executes an action. `python main.py run-agent-voice-debug`
shows the transcript and then uses the same explicit confirmation and safety
boundary as `run-agent-generic-debug`. The normalized transcript is passed
directly to that existing agent; voice does not rewrite or translate it.

The microphone adapter uses `sounddevice` for mono 16 kHz PCM capture and keeps
the WAV in memory only. OpenAI's recommended `gpt-transcribe` transcription
adapter uses the existing `OPENAI_API_KEY`; the audio is sent to OpenAI for transcription.
`VOICE_STT_TIMEOUT_SECONDS` defaults to 20 seconds and the SDK is configured for
at most one retry. Transcripts are trimmed and repeated whitespace is collapsed;
no language is forced. Raw audio is not written to disk or included in
diagnostics, and is released after transcription on success or failure.

Actions use fixed `kind` discriminators. Click targets are opaque IDs from the
latest observation, not coordinates or native Windows handles. Text entry is
literal and does not submit; key chords use an explicit allowlist and app IDs
must resolve through the trusted local catalog.
The executor accepts the complete `Action` union, handling `FinishAction` locally
without Windows calls. `ComputerAction` still names the desktop-only subset.

Dataclass type hints are static contracts, not runtime validation of model
responses. The Jev adapter validates the documented response and maps only
offered choices. The loop applies the baseline policy and an optional additional
policy before every action; confirmation-required policies are fail-closed when
no confirmation provider is configured. This remains a supervised, bounded
foundation rather than a general-purpose autonomous computer agent.

## Secrets and privacy

`.env.example` contains blank key placeholders and non-secret defaults. The CLI
loads the project-local `.env` without overriding existing process environment
variables. Keep credentials in local environment variables or an ignored `.env`;
the CLI does not log keys or file contents. Never put actual API keys in code,
tests, logs, or commits. `.gitignore` excludes environment files, key files,
recordings, observations, and logs. Review staged changes before committing;
ignore rules do not protect secrets pasted into tracked files.

## Scope and licensing

The Jev integration follows the official HTTP API and TYPESAFE_API_KEY convention.
No global shortcut library, TTS, OCR pipeline, real visual coordinate execution,
or free-form text generator is included. Real desktop coverage
depends on the target application and Windows session permissions.
An open-source license must be chosen before publishing; none is assumed here.
