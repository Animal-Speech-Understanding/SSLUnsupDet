# SSL Embedder of audios with sperm whale clicks

## Preparation for usage
1. Clone repository
    ```commandline
    git clone ...
    ```
2. Install [poetry](https://python-poetry.org/docs/) on your machine
3. Install dependencies
    ```commandline
    poetry install
    ```

## Train embedder
Run:
```commandline
python ssl/train.py -c pipeline.json
```

As result of training you can find model in the new folder named `ssl/Pipeline`

## Inference
...
