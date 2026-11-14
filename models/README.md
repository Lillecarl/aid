# aid-models

aid's own model clients as a library, for programs that want aid's embedders or rerankers without the daemon.
aid depends on it; it never imports aid.

```python
from aid_models.embedding import OpenAIEmbedder
from aid_models.transport import Provider

openrouter = Provider("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY")
embedder = OpenAIEmbedder("openrouter/qwen/qwen3-embedding-8b", openrouter, dimensions=1024)
vectors = (await embedder.embed(["def f(): ..."])).vectors
await embedder.aclose()
```
