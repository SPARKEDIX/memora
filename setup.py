from setuptools import setup

setup(
    name="memora",
    version="0.1.0",
    description="Smart LLM memory with dedup and hybrid search",
    author="Kartik",
    packages=["memora"],
    package_dir={"memora": "memora"},
    install_requires=[
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0",
        "numpy>=1.21.0",
        "rank-bm25>=0.2.2",
    ],
    python_requires=">=3.8",
)