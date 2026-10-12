# LiteLM agents

Aid owns the model/tool loop. LiteLM handles provider requests, translation, streaming, and embeddings.
Pydantic validates tool arguments, session specs, protocol messages, and typed final output.
Neither PydanticAI nor LiteLLM is a runtime dependency.

```python
import aid

reviewer = aid.Agent(
    "openrouter/deepseek/deepseek-chat",
    instructions="Review the changes and report defects.",
    toolsets=[aid.coding_tools],
)
```

Use `aid new-py reviewer my_agents:reviewer --permission ask` for an imported agent.
For catalog discovery, subclass `aid.AgentDefinition` and return `aid.Agent` from `build`.
Models use LiteLM's `provider/model` names. Provider credentials come from the worker environment.
The package includes Anthropic and OpenAI-compatible providers. Other providers need their optional SDK dependencies.
`model_settings` passes provider options, including `api_base`, `api_key`, `timeout`, and `num_retries`.
The default transport retry count is zero. Stream failures never replay completed output or tool effects.

Tools are Python functions or `aid.Tool` objects. Group them with `aid.Toolset`.
A tool can take `aid.RunContext` as its first argument to receive its tool-call identity.
Aid validates arguments before execution. Correctable failures raise `aid.ModelRetry`.
Unwrapped tool exceptions end the turn. Synchronous tools run in worker threads.
Tools run sequentially in provider order. Aid does not retry interrupted tools.

Pass a Pydantic model as `output_type` for validated final output.
`output_retries` bounds output correction requests. `max_requests` bounds all main-loop requests when configured.
An unset request budget is unlimited. Loop detection remains a separate supervisor signal.

Each turn saves available conversation parts and emits usage on success, failure, and cancellation.
Unavailable token counts remain unknown. Partial tool calls close as interrupted before resume.
Compaction changes only the stored prefix. Turn completion appends only that turn's messages.
Signed Anthropic thinking blocks remain intact for replay. Wire-only session metadata never enters stored history.
The Nix package patches LiteLM's Anthropic stream to retain cache usage, thinking-block identities, and redacted thinking.
It also closes each per-request Anthropic client. Worker shutdown closes LiteLM's cached OpenAI clients.

Stored `pydantic-ai` session kinds remain readable. Their workers use the LiteLM backend.
The aliases `aid.PydanticAISpec` and `aid.PydanticAgent` refer to the native spec and definition contract.
Agent modules must replace PydanticAI imports with `aid.Agent` and use slash-separated model names.
The backend reads supported legacy history parts without importing PydanticAI and retains their extra provider fields.
