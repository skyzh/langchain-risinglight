# risinglight-vector-connector

Use RisingLight as a vector store for LangChain.

## Usage

```python
git clone https://github.com/risinglightdb/risinglight
cd risinglight
python3 -m venv .venv
source .venv/bin/activate
cargo build --features python
git clone https://github.com/risinglightdb/risinglight-vector-connector.git
cd risinglight-vector-connector
pip install -r requirements.txt
python3 main.py
```