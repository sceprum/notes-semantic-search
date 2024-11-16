### Installation

Create and activate a new virtual environment
```bash
python3 -m venv env
source env/bin/activate
```

Install the required dependencies:
```python3 -m pip install -r requirements.txt```


### Usage

```bash
python3 search.py -v /path/to/directory/with/notes -n 3 'How to live a healthy life?'
```

Replace `/path/to/directory/with/notes` with the directory containing your notes.
Use `-n` to specify the number of results to return.

For more information on available command-line arguments, run:
```python3 search.py --help```

