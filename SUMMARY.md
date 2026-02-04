# Findings

02/04/2026:

After experimenting local LLM + RAG on both the Ubuntu and the Raspberry Pi, the conclusion on the bottleneck of both the Ubuntu iMac and the Pi 5 is hardware, specifically CPU-only LLM inference with limited memory bandwidth, not the RAG design or code.  The iMac Late 2012 (Ivy Bridge + DDR3) and Pi 5 (ARM + LPDDR4X) can both run local LLM + RAG, but:

- Embedding calls (nomic-embed-text) and generation (phi / llama / gemma) are large matrix multiplies over hundreds of MB of weights, which are memory-bandwidth bound on these CPUs.

- Typical result: 3–15 tokens/sec → 10–60 seconds per answer, with fans ramping and 100% CPU usage—exactly what you saw.

A small GPU board like the Jetson Orin Nano Super adds:

- Dedicated CUDA cores + higher effective memory bandwidth for LLM workloads.
- 15–25+ tokens/sec on 4B–8B quantized models at ~15–25 W, which moves you into usable, near-real-time RAG chat.

The architecture (Ollama + embeddings + Qdrant/Chroma + OER ingestion) is sound. The performance gap is fundamentally between CPU vs GPU on older/low-power hardware, and the Jetson class device is exactly the right next step to address it.
