# Setup

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
Finally, if you have the kaggle [CLI api installed](https://github.com/Kaggle/kaggle-api), then, inside kaggle_optiver, you can run

```
kaggle competitions download -c optiver-trading-at-the-close
```

After unzipping this, the project should be working

to get your kernel registered within vscode, while inside the virtual environment you may also need to install the proper kernel.

TO do this run
```
python3 -m ipykernel install --user --name=optiver_env
```
## Features

While we test our own features, many of our most successful features come from or are inspired by [this paper](https://www.nber.org/system/files/working_papers/w30366/w30366.pdf) on predicting short-term returns. While their paper studies this predictive ability in continuous markets, we find that many of these features also have strong predictive power for the closing cross.

