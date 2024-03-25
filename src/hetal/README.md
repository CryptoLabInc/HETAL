# HETAL


## Download and pre-preprocess datasets

Following 5 datasets will be used for experiments:

* MNIST
* CIFAR-10
* [DermaMNIST](https://www.nature.com/articles/s41597-022-01721-8)
* [Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)
* [SNIPS](https://arxiv.org/pdf/1805.10190v3.pdf)

*Except for Face Mask Detection* dataset, all the datasets will be downloaded and ViT-features will be extracted automatically under the following scripts.
The extracted features will be saved as numpy arrays (`*.npy` files) under the `data/{DATASET_NAME}` directory.

### Image datasets

For image datasets (MNIST, CIFAR-10, DermaMNIST), run the following command.

```
python data/vit.py --data DATASET_NAME --model MODEL_TYPE
```

Here are descriptions of the arguments:

| Argument  | Description |
|-----------|-------------|
| `--data DATASET_NAME`    | Dataset name. `mnist`, `cifar10`, `dermamnist`. |
| `--model MODEL_TYPE`    | Model size. `base`, `large`, or `huge`. Defaults to `base`. If you use `large` or `huge` models, then |


For each dataset, 6 `.npy` files will be stored in the corresponding directory: features (`features_train.npy`, `feature_val.npy`, `features_test.npy`) and labels (`label_train.npy`, `label_val.npy`, `label_test.npy`).

In case of `Face-Mask-Detection` dataset, you need to download raw dataset first from Kaggle page: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection.
Then put two directories `annotations` and `box-images` in the `face-mask-detection` directory.
one needs a further pre-processing step that crop faces and make them as individual images.
This can be done using `face_mask_detection.py`.

```
python data/face_mask_detection.py --model MODEL_TYPE
```

### SNIPS dataset
Run the following script. Then raw dataset will be downloaded and the extracted features (from MPNet) will be saved as `.npy` files as above.

```
python data/mpnet.py
```

## Fine-tune classifiers

Now let's run HETAL.
The extracted features will be encrypted and used for fine-tuning a classifier.

```
python run.py --data mnist --generate_keys --encrypted_train --early_stopping
```


| Argument  | Description |
|-----------|-------------|
| `--data DATASET_NAME`    | Dataset name. `mnist`, `cifar10`, `dermamnist`, `face-mask-detection`, `snips`, or their `_large` or `_huge` variants. |
| `--generate_keys`    | Generate HEaaN keys. Once you generate, you don't need to generate them again for further experiments. |
| `--encrypted_train`    | Whether to train classifier on encrypted dataset or not. |
| `--batch_size BATCH_SIZE`    | Batch size. |
| `--num_epoch NUM_EPOCH`    | Number of epochs to train. |
| `--lr LR`    | Learning rate. |
| `--sgd TYPE`    | Whether to use vanilla SGD (`vanilla`) or nesterov acceleration (`nesterov`). Defaults to `nesterov`. |
| `--early_stopping`    | Whether to use early stopping or not. |
| `--patience PATIENCE`   | Patience for early stopping. Defaults to 3. |
