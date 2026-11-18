# aid-agent

aid's agent loop as a library: `aid_agent.loop.run` runs one turn of a model with tools, run against Chat Completions, OpenAI's Responses API,
Anthropic's Messages API or Gemini, over aid's own HTTP clients (aid-models). aid depends on it; it never imports aid.
