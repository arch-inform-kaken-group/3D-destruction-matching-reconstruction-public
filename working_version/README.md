# 3D Destruction Reconstruction

Destruction of 3D models into fragments.

## SETUP

- SETUP Python 3.12 ENV [Python 3.12](https://www.python.org/downloads/release/python-31211/)

    ```
    python3.12 -m venv descon
    ```

    OR use search bar from VS Code

- Source ENV

    ```
    descon/Scripts/activate.bat
    ```

    OR set as interpreter from VS Code

- Install packages

    ```
    pip install -r requirements.txt
    ```

## USAGE

```bash
python ./src/destruction.py ./pottery --pieces 30 --randomness 0.5 --smoothness 0.9
```