# Corax Prompt Runtime

`prompts.runtime` assembles operator-editable Markdown into cache-stable model
messages. The first system message contains only static policy. Per-turn
identity, runtime context, selected skills, project instructions, recall, and
tool summaries live in a frozen hidden envelope. Tool discovery appends a new
envelope after the tool result; it never rebuilds the existing turn prefix.
Profile and working-memory edits therefore change only a future appended
envelope, never the eternal core system prefix.

The provider-native tool list is always the fixed `tool_search` + `tool_call`
pair. Schemas selected for a turn are runtime data inside the appended envelope
so changing capabilities cannot invalidate the provider prefix. Earlier
envelopes remain only in RAM as part of that prefix; the newest runtime block
governs the newest user turn. Traces expose IDs and hashes, never schemas,
compiled prompts, profile text, or recalled memory.

## Integration

```python
runtime = PromptRuntime()
runtime.bind(
    runtime_root=extension_root,
    data_root=data_root,
    workspace_root=workspace_root,
    legacy_prompt_root=legacy_root,  # optional
    config=agent_config.prompts,     # mapping, optional
)

request = await runtime.build(
    {
        "history": raw_history,
        "turn_messages": [{"role": "user", "content": user_text}],
        "selected_skills": selected_skills,
        "recalled_records": recalled_records,
        "project_path": project_path,
    },
    context={
        "_corax_prompt_context": {
            "channel": "console",
            "session_id": session_id,
            "turn_id": turn_id,
            "user_text": user_text,
            "tool_descriptors": active_tool_descriptors,
        }
    },
)
```

Call `build` again with the same session/turn and an append-only
`turn_messages` list after each tool result. Expanded descriptors are emitted
as a `tool_update` envelope after that result. Call `end_turn` with the final
assistant text. Effective history, including hidden envelopes and the tool
loop, is retained only in process RAM; normal disk history remains the user
message plus final assistant answer.

Supported operations through `ExtensionRequest` are `status`, `validate`,
`reload`, `migrate`, `build`, `retain_profile`, `identity`, and `end_turn`.

## Files and configuration

The 22 packaged defaults live in `prompts/`. On bind they are copied to missing
files below `<data_root>/<root>`. A byte-identical older stock default is
upgraded safely; any operator edit is preserved. Legacy
`system.md` and `safety.md` are imported only by the explicit `migrate`
operation, as dynamic compatibility layers; they never replace the stable core
prefix. Historical unmodified stock files are recognized and skipped so the
old monolith cannot be duplicated into the new prompt. A legacy profile is
migrated automatically. Identity defaults are
`<data_root>/identity/USER.md` and `MEMORY.md`.

Configuration keys are `enabled`, `root`, `user_profile`, `working_memory`,
`max_profile_chars`, `max_working_memory_chars`, `max_layer_chars`, and
`max_total_prompt_chars`. Matching `CORAX_PROMPTS_*` variables are fallback
values. Paths must remain inside `data_root`.

## Layer map and selection

The ordinary turn order is:

1. `core/SOUL.md`, `SYSTEM.md`, `PRINCIPLES.md`, `SAFETY.md`, then
   `behavior/RESPONSE_STYLE.md`;
2. the shared `identity/USER.md` profile and `identity/MEMORY.md` working
   snapshot;
3. generated `templates/RUNTIME_CONTEXT.md` and one channel layer;
4. selected behavior/service layers, safely wrapped recall, selected skills,
   then scoped project instructions;
5. session history and the current user message.

The five core/style files form the stable provider prefix. Runtime date,
identity, recall, tools, skills, project files, onboarding, corrections, and
channel guidance are appended as turn data. Selection is conditional:

| Turn | Additional layers |
| --- | --- |
| normal | `TOOL_USE`, channel, runtime, identity |
| current information | `CURRENT_INFORMATION` |
| first-run profile | `ONBOARDING` |
| tool failure | `RECOVERY` |
| corrected answer | generated `RETRACTION_NOTICE` |
| recalled memory | `MEMORY_RECALL` + `RELEVANT_MEMORY` wrapper |
| selected skill/project | scoped skill and `AGENTS.md` blocks |
| subagent | bounded `SUBAGENT` + `DELEGATED_TASK`; no full profile/history |
| heartbeat/scheduled | `HEARTBEAT`; no onboarding |
| compaction | `CONTEXT_COMPACTION` |

Templates also include `USER_PROFILE`, `WORKING_MEMORY`, and the wrappers
listed above. `MEMORY_RETENTION.md` documents the retention boundary; the host
still performs validation and persistence.

## Trust, identity, and persistence

- Core, runtime, and selected workflows are instructions. Runtime policy
  remains authoritative and cannot be granted or revoked by Markdown.
- `USER.md` contains durable, high-frequency user facts. `MEMORY.md` is a
  bounded always-visible working snapshot. Both are shared by Console, TUI,
  Telegram, and future channels.
- Semantic provider recall is separate, selected per turn, escaped, and
  enclosed as untrusted data. Current user statements and verified tool facts
  outrank it.
- Full prompt bundles, schemas, recall, and selected skill bodies are never
  written to conversation checkpoints or traces. They remain in process RAM
  only to preserve the append-only provider prefix; restart begins a cold cache
  from raw user/assistant history.
- A correction produces a bounded hash/category record for the host. On later
  turns the runtime rebuilds a model-only notice directly before the retracted
  assistant message; no message text or static system content enters that
  ledger.
- Context compaction establishes a new RAM cache epoch. Within a turn, file
  reloads and external mutation cannot alter the frozen snapshot.

## Operator guide

Use the active installation's runtime data directory, not the immutable source
checkout:

- change Corax's character in `runtime/data/prompts/core/SOUL.md`;
- change operational behavior in `core/SYSTEM.md` or `core/PRINCIPLES.md`;
- change answer style in `behavior/RESPONSE_STYLE.md`;
- inspect or correct the shared profile in `runtime/data/identity/USER.md`;
- edit short working context in `runtime/data/identity/MEMORY.md`;
- configure or clear semantic memory through the selected memory provider.

Keep `Onboarding-Complete: true` in a completed profile unless onboarding
should run again. Do not place credentials in prompt or identity files.

```sh
corax prompts status
corax prompts validate
corax prompts reload
corax prompts migrate
corax prompts identity status profile
corax prompts identity show profile
corax prompts identity replace profile ./USER.md
corax prompts identity reset memory
corax prompts identity onboarding profile
```

`status` reports hashes, sizes, and token estimates without printing private
contents. `validate` fails closed on broken required layers. `reload` affects
the next turn, never an active tool loop. `migrate` imports legacy `system.md`
and `safety.md` only as dynamic compatibility layers.

Identity `status` reports the active path and `chars/max` without content.
`show` is the only action that prints private identity text. `replace` reads
the named local file in the CLI and performs a bounded atomic replacement.
`reset profile` restarts onboarding without deleting semantic-memory backend
records; `onboarding profile` preserves current facts and only marks onboarding
incomplete.

To restore one packaged default, first move the operator file to a backup
outside `runtime/data/prompts`, then run `corax prompts reload`; the missing
file is recreated. Upgrades automatically replace only byte-identical older
stock defaults, so an operator-edited file is never silently overwritten.

## Test

```sh
PYTHONPATH=../agent-core:../agent-sdk python -m unittest discover -s tests -v
```
