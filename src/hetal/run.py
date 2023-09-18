import argparse
import timeit
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

import heaan_sdk
import heaan_sdk.ml as hml

import load_data
from hyperparams import get_lr, get_batch_size, get_num_epoch


def run(
    data: str,
    generate_keys: bool = False,
    encrypted_train: bool = True,
    batch_size: int = 128,
    num_epoch: int = 5,
    lr: float = 1.0,
    nesterov: bool = True,
    early_stopping: bool = True,
    patience: int = 3,
    with_gpu: bool = False,
):
    msg = f"dataset: {data}\n"
    msg += f"encrypt train: {encrypted_train}\n"
    msg += f"batch size: {batch_size}\n"
    msg += f"epochs: {num_epoch}\n"
    msg += f"learning rate: {lr}\n"
    msg += f"nesterov: {nesterov}\n"
    msg += f"use early stopping: {early_stopping}\n"
    msg += f"patience: {patience}\n"
    msg += f"use gpu: {with_gpu}\n"
    print(msg)

    # fix initial seed
    np.random.seed(0)

    # setup
    print("Setup: generate keys, build context, ...")
    key_dir_path = Path("keys")
    params = heaan_sdk.HEParameter.from_preset("FGb")
    context = heaan_sdk.Context(
        params,
        key_dir_path=key_dir_path,
        load_keys="all",
        generate_keys=generate_keys,
        make_bootstrappable=True,
        use_gpu=with_gpu,
    )

    # load data
    print(f"Load {data} data ...")
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = load_data.load_data(data)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # both unit_shape[0] and unit_shape[1] should be larger than or equal to
    # (padded) number of classes
    pad_num_classes = 2 ** np.ceil(np.log2(num_classes))
    s0 = min(max(batch_size, pad_num_classes), context.num_slots // pad_num_classes)
    unit_shape = (s0, context.num_slots // s0)
    classes = list(map(str, range(num_classes)))
    num_feature = len(X_train[0])

    # preprocessing
    print("Preprocessing ...")
    train_data = hml.preprocessing.encode_train_data(
        context,
        X_train,
        y_train,
        unit_shape,
    )
    val_data = hml.preprocessing.encode_train_data(
        context,
        X_val,
        y_val,
        unit_shape,
    )
    if encrypted_train:
        print("Encrypt train and validation data")
        train_data.encrypt()
        val_data.encrypt()
    else:
        print("Run without encryption.")

    # train
    print("Set model")
    model = hml.Classifier(context, unit_shape, num_feature, classes)
    st = timeit.default_timer()
    if encrypted_train:
        print("Encrypt model")
        model.encrypt()
    if with_gpu:
        model.to_device()

    print("Train a model with HETAL")
    model.fit_val_loss(
        train_data_set=train_data,
        val_data_set=val_data,
        lr=lr,
        num_epoch=num_epoch,
        batch_size=batch_size,
        nesterov=nesterov,
        early_stopping=early_stopping,
        patience=patience,
    )
    if with_gpu:
        model.to_host()
    if model.encrypted:
        model.decrypt()
    et = timeit.default_timer()
    train_time = et - st

    epoch = model.epoch_state if early_stopping else num_epoch
    print(f"Train takes {train_time:.2f} seconds for total {epoch} epochs.")
    print(f"Per step: {train_time / (model.step_state - 1):.2f}s")
    if early_stopping:
        print(f"Epoch with minimum validation loss: {model.best_epoch}")

    print(f"DiagABT cost: {context.abt_time:.2f}s, DiagATB cost: {context.atb_time:.2f}s")
    matmul_time = context.abt_time + context.atb_time
    print(f"Total matmul cost: {matmul_time / train_time * 100:.2f}%")

    model.train_mode = False
    print(model)
    print("validation losses", model.val_losses)

    # test
    X = X_test
    y = y_test

    test_data_feature = heaan_sdk.HEMatrix.encode(context, X, unit_shape)
    output = model.predict(test_data_feature)
    output_arr = output.decode()

    # accuracy
    preds = output_arr.argmax(axis=1)
    correct_cnt = (preds == y).sum()
    acc_ = correct_cnt / len(y)
    print(f"Test accuracy: {acc_ * 100: .2f}%")

    # classification report
    print(classification_report(y, preds, digits=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HETAL")
    parser.add_argument("--data", type=str, choices=load_data.DATA_LIST, required=True)
    parser.add_argument("--generate_keys", action="store_true")
    parser.add_argument("--encrypted_train", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--sgd", type=str, choices=["vanilla", "nesterov"], default="nesterov")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--with_gpu", action="store_true")

    args = parser.parse_args()

    # If hyperparameters are not provided, use default values.
    if args.lr is None:
        args.lr = get_lr(args.data)
    if args.batch_size is None:
        args.batch_size = get_batch_size(args.data)
    if args.num_epoch is None:
        args.num_epoch = get_num_epoch(args.data)
    
    # nesterov acceleration
    nesterov = True if args.sgd == "nesterov" else False

    run(
        data=args.data,
        generate_keys=args.generate_keys,
        encrypted_train=args.encrypted_train,
        batch_size=args.batch_size,
        num_epoch=args.num_epoch,
        lr=args.lr,
        nesterov=nesterov,
        early_stopping=args.early_stopping,
        patience=args.patience,
        with_gpu=args.with_gpu,
    )
