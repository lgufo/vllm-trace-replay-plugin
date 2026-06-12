from setuptools import find_packages, setup


setup(
    name="vllm-trace-replay-plugin",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    # vLLM: do not pin to a PyPI version many mirrors never ship (e.g. 0.13+).
    # This plugin targets v1 WorkerBase APIs; use a recent vLLM when possible.
    install_requires=[
        "vllm>=0.6.0",
        "torch>=2.0.0",
        "transformers>=4.40.0",
    ],
    entry_points={
        "vllm.platform_plugins": [
            "trace_replay = trace_replay_plugin:trace_replay_platform_plugin",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
