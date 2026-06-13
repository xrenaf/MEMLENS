# MEMLENS Memory-Agent Reproduction Notes

This folder documents the public-facing setup used to evaluate memory-augmented
agent baselines on MEMLENS. It is intentionally not a full copy of the complete
benchmark runners. The goal is to make the data conversion, prompt construction,
retrieval settings, and model choices explicit enough for reproducibility.

The memory-agent experiments use the fixed 195-question agent subset described
in the main repository README. Each benchmark item contains one question and a
set of MEMLENS haystack sessions. For every memory-agent baseline, one MEMLENS
item is converted into one framework-specific evaluation sample.

## Files

- `prompt_builders.py`: prompt builders for the methods where the MEMLENS
  evaluation layer used a lightweight custom prompt.
- `README.md`: this reproduction note.

## Shared Evaluation Convention

Unless stated otherwise, sessions are consumed in the original MEMLENS order.
The original `haystack_sessions` field is treated as the per-question memory
source, while the target question and its date are used only at retrieval and
answer time. Images are either preserved, rendered, or stripped depending on the
capability expected by each memory-agent framework.

The final judge prompt used to score answers is separate from the agent prompts
below and is not part of this folder.

## Prompt Provenance

| Method | MEMLENS-side custom prompt? | Where it is used | Public helper |
| --- | --- | --- | --- |
| mem0 | Yes | Final answer over retrieved mem0 memories | `build_mem0_answer_messages()` |
| Memory-T1 | Yes | Final answer after BM25 session retrieval and temporal filtering | `build_memory_t1_prompt()` |
| M3C | Yes | QA generation over top retrieved sessions | `build_m3c_qa_messages()` |
| M3-Agent | Yes, for memorization only | Rendered-session image to text memory | `build_m3_agent_image_memory_prompt()` |
| M2A | No | Uses the upstream M2A evaluator and agent/tool prompts | N/A |
| MemOS | No | Uses the upstream MemOS LongMemEval-style ingestion, search, and response scripts | N/A |
| MemAgent-7B | No | Uses the upstream recurrent memory-agent runner | N/A |

## mem0

Recommended package version: `mem0ai==2.0.4`.

MEMLENS does not use `Memory.chat()` for this baseline. The evaluation wrapper
uses mem0 as a memory construction and search component, then performs the final
answer step with an OpenAI-compatible chat completion over retrieved memories.

Per-question procedure:

1. Create a fresh `mem0.Memory` instance for the current question.
2. Add all MEMLENS haystack sessions for that question in original order.
3. Strip image payloads from messages and replace inline image markers with
   `[image]` for text-only memory construction.
4. Prefix each message with its session date and also pass `session_id` and
   `session_date` as metadata.
5. Search the per-question memory store with the benchmark question.
6. Build the final answer messages with `build_mem0_answer_messages()`.
7. Call an OpenAI-compatible chat completion with `temperature=0.0` and
   `max_tokens=2048`.

Key settings:

| Setting | Value |
| --- | --- |
| mem0 version | `mem0ai==2.0.4` |
| mem0 config version | `v1.1` |
| Vector store | FAISS, one fresh store per question |
| Distance metric | Cosine |
| FAISS normalization | `normalize_L2=True` |
| Search top-k | `top_k=20` |
| Memory filters | `{"user_id": question_id}` |
| Default LLM | `gpt-4.1-mini-2025-04-14` |
| Default embedding model | `text-embedding-3-small` |
| Answer decoding | `temperature=0.0`, `max_tokens=2048` |

Equivalent search call:

```python
memories = memory.search(
    query=question,
    filters={"user_id": question_id},
    top_k=20,
)
```

If a different embedding model is used, make sure the FAISS dimension matches
the embedding output dimension.

## Memory-T1

Memory-T1 is evaluated as a text-only memory baseline.

Data conversion:

- Convert each MEMLENS item into a package containing the question and a list of
  labeled dialogue sessions.
- Keep the original session order and utterance dates.
- Remove image references from the dialogue text.

Retrieval and answering:

- Retrieve candidate sessions with BM25.
- Use `bm25_top_k=10`.
- Apply the temporal filter used by the Memory-T1-style evaluation path.
- Pack retrieved sessions under the model prompt budget.
- Build the final prompt with `build_memory_t1_prompt()`.

Key settings:

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Retrieval | BM25 over sessions |
| Retrieval top-k | `10` |
| Temporal filter | Enabled |
| Prompt budget | Approximately 30k prompt tokens for a 32768-token model context |
| Answer format | Last line should be `Answer: <answer>` |

## M3C

M3C is evaluated as a retrieval-then-QA baseline over MEMLENS sessions.

Data conversion:

- Keep MEMLENS sessions as session-level records.
- Use text-only session encoding for retrieval.
- Preserve the original order before retrieval; the QA prompt receives sessions
  in retrieved rank order.

Retrieval and answering:

- Encode each session with the M3C retrieval model.
- Use the first 4 turns of each session for retrieval encoding.
- Retrieve the top sessions and pass them to a concise QA prompt.
- Build the QA chat messages with `build_m3c_qa_messages()`.

Key settings:

| Setting | Value |
| --- | --- |
| Retrieval model | `jihyoung/M3C-retrieval` |
| Generation base model | `Qwen/Qwen2-VL-2B-Instruct` |
| Retrieval top-k | `3` |
| Max turns per retrieved session in QA context | `20` |
| Max QA context length | `12000` characters |
| Image setting | `--no_images` for the reported text-only M3C run |

## M3-Agent

M3-Agent is evaluated with rendered MEMLENS sessions so that both conversation
text and embedded images can be observed during memorization.

Data conversion:

- Render each MEMLENS session as a screenshot-style image.
- Each rendered image contains the session text and embedded visual content.
- Use the rendered images as memorization inputs.

Memorization:

- Use the rendered-session image memorization prompt in
  `build_m3_agent_image_memory_prompt()`.
- The memorization model produces text memories with two fields:
  `video_descriptions` and `high_level_conclusions`.
- Store generated memories in a per-question graph/index.

Control and answering:

- Use the upstream M3-Agent control loop and action format.
- The model emits either `Action: [Search] Content: <query>` or
  `Action: [Answer] Content: <answer>`.
- Search actions retrieve memories from the per-question memory graph and append
  the retrieved memory text back into the conversation.

Key settings:

| Setting | Value |
| --- | --- |
| Memorization model family | M3-Agent memorization / Qwen2.5-Omni-style model |
| Rendered-session batch size | `10` |
| Memory embedding model | `text-embedding-3-large` |
| Control model family | M3-Agent control / QwQ-32B-style model |
| Max control rounds | `5` |

## M2A

M2A is evaluated through its upstream memory construction, retrieval, tool-use,
and answering pipeline. MEMLENS only provides the dataset conversion and serving
configuration.

Data conversion:

- Convert each MEMLENS item into a LoCoMo-like sample.
- Keep sessions in original order as `session_0`, `session_1`, and so on.
- Map the two dialogue roles to the speaker IDs expected by M2A.
- Use dialogue IDs such as `S<session_index>:<turn_index>`.
- Preserve image references as local paths or URLs when available.

Tool calls:

- Qwen tool-call conversion is handled by the vLLM OpenAI-compatible server.
- Start vLLM with `--enable-auto-tool-choice --tool-call-parser qwen3_xml`.
- The evaluator receives OpenAI-style `tool_calls` from the server.

Embedding services:

| Embedding type | Model / service |
| --- | --- |
| Text embedding | `sentence-transformers/all-MiniLM-L6-v2` served with vLLM embedding mode |
| Image embedding | SigLIP2 / `SigLIP2-so400m` service |

## MemOS

MemOS is evaluated with the upstream LongMemEval-style ingestion, search, and
response stages.

Data conversion:

- Convert MEMLENS items into LongMemEval-like JSON records.
- Use a text-only representation.
- Remove inline image markers from the memory text.

Key settings:

| Setting | Value |
| --- | --- |
| Pipeline stages | Ingestion, search, response |
| Search top-k | `20` |
| Search mode | Fast |
| Workers | `10` |
| Prompt provenance | Upstream MemOS response prompt |

## MemAgent-7B

MemAgent-7B is evaluated with the upstream recurrent memory-agent runner rather
than a top-k retrieval wrapper.

Data conversion:

- Sort sessions by `session_date`.
- Flatten the text-only sessions into one long context.
- Remove image markers from the flattened context.

Key settings:

| Setting | Value |
| --- | --- |
| Model | `BytedTsinghua-SIA/RL-MemoryAgent-7B` |
| Retrieval style | Recurrent chunk-based memory, not top-k retrieval |
| Recurrent chunk size | `5000` tokens |
| Max recurrent context length | `300000` tokens |
| Max new tokens | `1024` |
| vLLM max model length | `16384` |
| Prompt provenance | Upstream MemAgent runner |

## Summary Table

| Method | Session handling | Image handling | Retrieval / memory setting | Embedding model | Answer prompt |
| --- | --- | --- | --- | --- | --- |
| mem0 | Original order, add all sessions to a fresh per-question memory | Text-only, `[image]` markers | `top_k=20` | `text-embedding-3-small` | MEMLENS custom prompt |
| Memory-T1 | Original order, labeled sessions | Text-only | BM25 `top_k=10` + temporal filter | BM25 lexical retrieval | MEMLENS custom Memory-T1-style prompt |
| M3C | Original sessions, QA receives retrieved rank order | Text-only for reported run | `top_k=3` | `jihyoung/M3C-retrieval` | MEMLENS custom concise QA prompt |
| M3-Agent | Rendered session images | Rendered text + images | Iterative memory search in M3-Agent control loop | `text-embedding-3-large` | Upstream control prompt; MEMLENS memorization prompt |
| M2A | LoCoMo-like sessions in original order | Preserved paths/URLs | Upstream M2A memory pipeline | `all-MiniLM-L6-v2` text, SigLIP2 image | Upstream M2A prompt |
| MemOS | LongMemEval-like text records | Text-only | `top_k=20`, fast search | Upstream MemOS setting | Upstream MemOS prompt |
| MemAgent-7B | Date-sorted flattened context | Text-only | Recurrent 5000-token chunks | N/A | Upstream MemAgent prompt |
