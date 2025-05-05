# SSL Embedder of audios with sperm whale clicks & Click Detection

## Preparation for usage
1. Clone repository
    ```bash
    git clone ...
    ```
2. Install [poetry](https://python-poetry.org/docs/) on your machine
3. Install dependencies
    ```bash
    poetry install
    ```

## Train embedder
1. Activate environment with:
   ```bash
   eval $(poetry env activate)
   ```
   Or skip and use `poetry run ...`
2. Run:
   ```bash
   python ssl/train.py -c pipeline.json
   ```
   or 
   ```bash
   poetry run python ssl/train.py -c pipeline.json
   ```
3. Check tensorboard in `Pipeline/logs`

As result of training you can find model in the new folder named `Pipeline/models/final_model.pth`

## Train Click Detector
1. Run all cells in notebooks `train-lstm-detector.ipynb` or `train-cnn-detector.ipynb`
2. During running you may specify path to embedder model or parameters of detector model.
3. Check tensorboard logs in `lstm-detector` or `cnn-detector` folder
