LEARNING_RATE = {
    "mnist": 2.0,
    "mnist_large": 0.05,
    "cifar10": 1.0,
    "cifar10_large": 0.1,
    "face-mask-detection": 0.5,
    "face-mask-detection_large": 0.1,
    "dermamnist": 0.3,
    "dermamnist_large": 0.03,
    "snips": 1.0,
}

BATCH_SIZE = {
    "mnist": 1024,
    "mnist_large": 1024,
    "cifar10": 2048,
    "cifar10_large": 2048,
    "face-mask-detection": 512,
    "face-mask-detection_large": 512,
    "dermamnist": 1024,
    "dermamnist_large": 1024,
    "snips": 1024,
}

NUM_EPOCH = {
    "mnist": 7,
    "mnist_large": 7,
    "cifar10": 9,
    "cifar10_large": 7,
    "face-mask-detection": 22,
    "face-mask-detection_large": 10,
    "dermamnist": 23,
    "dermamnist_large": 15,
    "snips": 14,
}


def get_lr(data: str) -> float:
    return LEARNING_RATE[data]

def get_batch_size(data: str) -> int:
    return BATCH_SIZE[data]

def get_num_epoch(data: str) -> int:
    return NUM_EPOCH[data]
