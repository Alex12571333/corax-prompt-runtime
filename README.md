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
`reload`, `migrate`, `build`, `retain_profile`, and `end_turn`.

## Files and configuration

The 22 packaged defaults live in `prompts/`. On bind they are copied to missing
files below `<data_root>/<root>` and never overwrite operator edits. Legacy
`system.md` and `safety.md` are imported only by the explicit `migrate`
operation, as dynamic compatibility layers; they never replace the stable core
prefix. A legacy profile is migrated automatically. Identity defaults are
`<data_root>/identity/USER.md` and `MEMORY.md`.

Configuration keys are `enabled`, `root`, `user_profile`, `working_memory`,
`max_profile_chars`, `max_working_memory_chars`, `max_layer_chars`, and
`max_total_prompt_chars`. Matching `CORAX_PROMPTS_*` variables are fallback
values. Paths must remain inside `data_root`.

## Test

```sh
PYTHONPATH=../agent-core:../agent-sdk python -m unittest discover -s tests -v
```
