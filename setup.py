from setuptools import setup, find_packages

setup(
    name="intersr",
    version="1.0.0",
    description="InterSR: Interleaved System-1/2 Reasoning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.37",
        "sglang>=0.2",
        "sentence-transformers",
        "scikit-learn",
        "stanza",
        "datasets",
        "math-verify",
        "pyyaml",
        "jinja2",
        "tqdm",
        "filelock",
    ],
)
