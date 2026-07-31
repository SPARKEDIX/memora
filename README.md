# Memora

Smart memory for LLMs. Auto-dedup, hybrid search, domain-aware.

## Install

```bash
pip install .

## Quick start
import memora

memory = memora.Memory(name="my_bot")
memory.add("Mujhe diabetes hai", user="rahul")
memory.add("Main walk karta hoon", user="rahul")

context = memory.get("Walk ke baad kya khaana?", user="rahul")
print(context)

## API
memory.add(text, user, ttl="30d") — Store memory
memory.get(query, user, top_k=5) — Retrieve context
memory.delete(user, domain) — Delete memories
memory.optimize() — Merge similar memories
memory.info() — Stats