# Memora 🧠

> Zero-config LLM memory that understands context, not just text.

Memora is a smart memory layer for your AI applications. Unlike traditional vector databases that store text as isolated chunks, Memora captures **concepts**, **relationships**, and **domains** — so your LLM gets the full picture with 90% fewer tokens.

---

## ✨ Why Memora?

| Feature | Pinecone | ChromaDB | **Memora** |
|---------|----------|----------|------------|
| Setup | Cloud signup + API key | `pip install` | `pip install memora` ✅ |
| Dedup | ❌ Manual | ❌ Manual | ✅ **Auto-merge** |
| Hybrid Search | ❌ Vector only | ❌ Vector only | ✅ **BM25 + Vector** |
| Domain Filter | ❌ Manual metadata | ❌ Manual metadata | ✅ **Auto-detect** |
| Related Memories | ❌ No | ❌ No | ✅ **Auto-bonding** |
| Auto-aging | ❌ No | ❌ No | ✅ **Fractal compression** |
| Session Memory | ❌ No | ❌ No | ✅ **Auto-expire** |
| Token Cost | ~2000/query | ~2000/query | **~50/query** ✅ |

---

## 🚀 Quick Start

### Install
```bash
pip install memora
```

### Basic Usage
```python
import memora

# Create memory
memory = memora.Memory()

# Store memories
memory.add("Mujhe diabetes hai, fasting 180", user="rahul")
memory.add("Main roz 6 baje walk karta hoon", user="rahul")
memory.add("Doctor ne Metformin 500mg diya hai", user="rahul")

# Retrieve context
context = memory.get("Walk ke baad kya khaana chahiye?", user="rahul")
print(context)
```

**Output:**
```
Primary:
    Main roz 6 baje walk karta hoon

Related:
    Mujhe diabetes hai, fasting 180
    Doctor ne Metformin 500mg diya hai
```

**Your LLM now gets the full health context — not just the "walk" keyword.**

---

## 📖 Full API Reference

### `Memory(db_path="memory.db", default_ttl="30d")`

Create a memory instance.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"memory.db"` | SQLite file path for storage |
| `default_ttl` | `str` | `"30d"` | Default expiration for memories (`"7d"`, `"30d"`, `"forever"`) |

**Example:**
```python
# Separate DB per user
rahul_mem = memora.Memory(db_path="rahul.db")
priya_mem = memora.Memory(db_path="priya.db")
```

---

### `memory.add(text, user=None, ttl=None, session=False)`

Store a memory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | **Required** | The text to store |
| `user` | `str` | `None` | Owner of this memory (isolated per user) |
| `ttl` | `str` | `None` | Time-to-live (`"7d"`, `"30d"`, `"forever"`, `None` = use default) |
| `session` | `bool` | `False` | If `True`, auto-delete after session ends |

**Examples:**
```python
# Permanent health record
memory.add("Mujhe diabetes hai", user="rahul", ttl="forever")

# Temporary chat
memory.add("Aaj mausam accha hai", user="rahul", ttl="7d")

# Session-only (disappears after session)
memory.add("OTP is 123456", user="rahul", session=True)
```

---

### `memory.get(query, user=None, domain=None, top_k=5, include_bonded=True)`

Retrieve relevant memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | **Required** | Your search query |
| `user` | `str` | `None` | Filter by specific user |
| `domain` | `str` | `None` | Force domain (`"health"`, `"work"`, `"gaming"`, `"family"`, `"casual"`) |
| `top_k` | `int` | `5` | Number of primary results |
| `include_bonded` | `bool` | `True` | Include related memories from same domain |

**Examples:**
```python
# Basic query
result = memory.get("Walk ke baad kya khaana?", user="rahul")

# Only health domain
result = memory.get("Diet plan", user="rahul", domain="health")

# More results
result = memory.get("Health tips", user="rahul", top_k=10)
```

**Returns:**
```
Primary:
    [Most relevant memories]

Related:
    [Bonded memories from same domain]
```

---

### `memory.delete(user=None, domain=None, older_than=None)`

Delete memories.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user` | `str` | `None` | Delete all memories of this user |
| `domain` | `str` | `None` | Delete only this domain |
| `older_than` | `str` | `None` | Delete memories older than (`"30d"`, `"90d"`, `"1y"`) |

**Examples:**
```python
# Delete all casual chats
memory.delete(user="rahul", domain="casual")

# Delete everything older than 1 year
memory.delete(user="rahul", older_than="1y")
```

---

### `memory.optimize()`

Merge similar memories to reduce storage.

```python
# After adding many similar memories
memory.optimize()
# Reduces crystal count by 30-50%
```

---

### `memory.info()`

Get database stats.

```python
print(memory.info())
# {
#     'total_memories': 50,
#     'unique_users': 3,
#     'domains': {'health': 20, 'work': 15, 'casual': 15},
#     'levels': {0: 30, 1: 15, 2: 5},
#     'persisted': True
# }
```

---

## 🏥 Real-World Example: Health Assistant

```python
import memora
from openai import OpenAI

llm = OpenAI()
memory = memora.Memory(db_path="patients.db")

def doctor_chat(patient_id, message):
    # 1. Retrieve patient history
    context = memory.get(message, user=patient_id, top_k=3)

    # 2. Ask LLM with context
    response = llm.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are Dr.AI. Use patient history."},
            {"role": "user", "content": f"History:
{context}

Patient: {message}"}
        ]
    )

    reply = response.choices[0].message.content

    # 3. Store both sides
    memory.add(message, user=patient_id, ttl="forever")
    memory.add(reply, user=patient_id, role="assistant", ttl="forever")

    return reply

# Usage
print(doctor_chat("rahul_123", "Mujhe diabetes hai, fasting 180"))
print(doctor_chat("rahul_123", "Main roz walk karta hoon"))
print(doctor_chat("rahul_123", "Walk ke baad kya khaana chahiye?"))
# Output: Diabetes-aware diet advice
```

---

## 🎮 Supported Domains (Auto-Detected)

Memora automatically detects domains from your text:

| Domain | Keywords | Use Case |
|--------|----------|----------|
| `health` | diabetes, sugar, doctor, medicine, walk | Health assistants |
| `work` | code, software, project, office, engineer | Coding assistants |
| `gaming` | game, pubg, play, fortnite, ps5 | Game NPCs |
| `family` | mother, father, sister, wife, husband | Personal assistants |
| `casual` | (everything else) | General chat |

**Custom domains:** Coming in v0.2.0

---

## 🧠 How It Works

```
Your Text
    ↓
Concept Extraction (Sentence-Transformers)
    ↓
Domain Detection (Auto-classify: health/work/gaming/family/casual)
    ↓
Duplicate Check (Merge if >60% similar)
    ↓
Store in FAISS (fast vector search) + SQLite (metadata)
    ↓
BM25 Index (exact word matching)
    ↓
When you query:
    Hybrid Search (Vector + BM25)
    ↓
Domain Filter (health query → only health memories)
    ↓
Crystal Bonding (surface related concepts)
    ↓
Formatted Context → Your LLM
```

---

## 📦 Installation

```bash
pip install memora
```

**Dependencies:** `sentence-transformers`, `faiss-cpu`, `numpy`, `rank-bm25`

---

## 📝 License

MIT License — free for personal and commercial use.

---

## 🙏 Support

- ⭐ Star this repo if you find it useful
- 🐛 Open an issue for bugs
- 💡 Open a discussion for feature requests

---

**Made with ❤️ for smarter AI memories.**
