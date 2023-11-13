# kaggle-optiver

must have [poetry](https://python-poetry.org/) installed.

to create a virtual environment run

```
python3.10 -m venv optiver_env
source optiver_env/bin/activate
```

and to install packages run

```
poetry lock
poetry install
```
Finally, if you have the kaggle [CLI api installed](https://github.com/Kaggle/kaggle-api), then you can run

```
kaggle competitions download -c optiver-trading-at-the-close
```

After unzipping this, the project should be working

to get your kernel registered within vscode, while inside the virtual environment you may also need to install the proper kernel.

TO do this run
```
python3 -m ipykernel install --user --name=optiver_env
```
