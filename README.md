# risinglight-vector-connector

Use RisingLight as a vector store for LangChain.

## Usage

```python
# Activate your virtual environment, i.e., using poetry
poetry new risinglight-test-environment
cd risinglight-test-environment
poetry add --dev maturin
poetry add git+https://github.com/skyzh/langchain-risinglight
# Install risinglight in the virtual env
poetry shell
git clone https://github.com/risinglightdb/risinglight
cd risinglight
maturin build -F python
pip install ./target/wheels/risinglight-*.whl --force-reinstall
```

## Local Development

```python
git clone https://github.com/skyzh/langchain-risinglight
cd langchain-risinglight
poetry env use 3.12
poetry install
poetry shell
git clone https://github.com/risinglightdb/risinglight ../risinglight
cd ../risinglight
maturin build -F python
pip install ./target/wheels/risinglight-*.whl --force-reinstall
cd ../langchain-risinglight
poetry run python example.py
```
