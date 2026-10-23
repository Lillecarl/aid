# LiteLM agents

Aid owns the model/tool loop and the message translation. The OpenAI and Anthropic SDKs carry the requests.
Pydantic validates tool arguments, session specs, protocol messages, and typed final output.
Neither PydanticAI, LiteLLM nor LiteLM is a runtime dependency; the backend keeps the `litelm` name.

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
Models use `provider/model` names (`src/aid/model.py`). `anthropic/` goes to the Messages API through the
Anthropic SDK. Every other provider goes to Chat Completions through the OpenAI SDK: `openai`, `deepseek`,
`openrouter`, `groq`, `gemini`, `mistral`, `xai`, `together`, `fireworks`, and the local `ollama`, `vllm` and
`lmstudio`. Another prefix works with an explicit `api_base`. A bare name is OpenAI's.
Credentials come from the worker environment (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`, ...)
or `api_key`. A missing key fails the request and names the variable.
`model_settings` configures the client with `api_key`, `api_base` (or `base_url`) and `max_retries` (or
`num_retries`). Every other key goes into each request: `extra_headers`, `extra_body`, `timeout`,
`temperature`, Anthropic's `thinking`, `max_tokens`, `output_config`, `cache_control`, and so on.
The default retry count is zero and the default timeout 600 s. Stream failures never replay completed output or
tool effects.
Anthropic requests default to `max_tokens` 32000 (4096 for `claude-3*`) and to automatic prompt caching
(`cache_control: {"type": "ephemeral"}`). Beta features need only their `anthropic-beta` header in
`extra_headers`.

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
Signed and redacted Anthropic thinking blocks remain intact for replay; thinking from another provider is not
sent to Anthropic. Chat Completions providers get reasoning text back as `reasoning_content`.
Wire-only session metadata never enters stored history.
Each model keeps one SDK client per configuration. Worker shutdown closes them.

Stored `pydantic-ai` session kinds remain readable. Their workers use the LiteLM backend.
The aliases `aid.PydanticAISpec` and `aid.PydanticAgent` refer to the native spec and definition contract.
Agent modules must replace PydanticAI imports with `aid.Agent` and use slash-separated model names.
The backend reads supported legacy history parts without importing PydanticAI and retains their extra provider fields.
