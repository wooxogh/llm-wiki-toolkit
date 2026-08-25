# Third-Party Models and Data

LLM-Wiki V3 source code is licensed under the MIT License in `LICENSE`.

The distribution includes `small_v3_50_normal.pt`, a 7.4 MB inference-only
checkpoint trained for the Small Attention boundary verifier. Its SHA-256 is:

```text
2887a57e41ba9c832f4562099b4a824ec68af9dd43c1c782d17acfb5d205b4e8
```

The checkpoint was trained from cached sentence representations produced by
`Qwen/Qwen3-Embedding-0.6B` on a Wiki-727K training subset. It does not bundle
the Qwen model, tokenizer, Wiki-727K documents, or dataset text.

- Qwen3 Embedding 0.6B: Apache-2.0, downloaded at first use from
  <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B>
- Wiki-727K dataset: MIT, source information at
  <https://huggingface.co/datasets/saeedabc/wiki727k>
- Optional BGE reranker: Apache-2.0, downloaded only when reranking is used,
  from <https://huggingface.co/BAAI/bge-reranker-v2-m3>

Users are responsible for reviewing the licenses and usage terms of downloaded
models and of the Markdown content they index.
