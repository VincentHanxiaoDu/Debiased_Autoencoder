import os
import requests
import logging
from Settings.settings import *
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

logger = logging.getLogger("dataset_utils")
logger.setLevel(LOG_LEVEL)


def download_file(url: str, file_path: str, buffer_size: int = 100000):
    """Download file at <url> to <file_path>.
    Parameters
    ----------
    url : str
        The url of the file.
    file_path : str
        The local absolute path of the file.
    buffer_size : int, optional
        The buffer size/chunk size in bytes of download,
        by default 100000.

    Returns
    -------
    str
        The absolute file path of the downloaded file.
    """
    # clear tqdm instances.
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()
    logger.info(
        f"Start downloading file from url: {url}.")
    logger.info(f"Download path: {file_path}.")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progression_bar = tqdm(total=total_size_in_bytes, unit='iB',
                               unit_scale=True)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=buffer_size):
                progression_bar.update(len(chunk))
                f.write(chunk)
            progression_bar.close()
    logger.info("File downloaded successfully!")
    return file_path


def download_CelebA(dir: str = DATASETS_DIR, filename: str = "CelebA.h5"):
    """Download the CelebA dataset.

    Parameters
    ----------
    dir : str, optional
        The absolute directory path to store the local file, by default
        <DATASETS_DIR>.
    filename : str, optional
        The file name of the dataset file, by default "CelebA.h5"

    Returns
    -------
    str
        The absolute file path of the dataset file.
    """
    # training dataset:
    # https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1
    celebA_url = "https://www.dropbox.com/s/hlz8atheyozp1yx/train_face.h5?dl=1"
    return download_file(celebA_url, os.path.join(dir, filename))


def generate_training_testing_indices(data_size: int,
                                      training_ratio: float = 0.8):
    """Randomly generated indices of training data and testing data.

    Parameters
    ----------
    data_size : int
        The dataset size.
    training_ratio : float, optional
        The ratio of training data size over all data size, must between 0
        and 1, by default 0.8.

    Returns
    -------
    tuple
        A tuple contains training indices and testing indices.
    """
    training_indices = generate_random_training_indices(data_size,
                                                        training_ratio)
    testing_indices = generate_testing_indices(data_size, training_indices)
    return training_indices, testing_indices


def generate_random_training_indices(data_size: int,
                                     training_ratio: float = 0.8):
    """Randomly generated sorted indices of the training data.

    Parameters
    ----------
    data_size : int
        The dataset size.
    training_ratio : float
        The ratio of training data size over all data size, must between 0
        and 1, by default 0.8.

    Returns
    -------
    np.ndarray
        A 1-d list contains all training indices.
    """
    assert 0 < training_ratio < 1
    return np.sort(np.random.choice(data_size,
                                    round(data_size*training_ratio),
                                    replace=False))


def generate_testing_indices(data_size: int, training_indices: list):
    """Generate sorted testing indices according to <training_indices>.

    Parameters
    ----------
    data_size : int
        the size of the entire dataset.
    training_indices : list
        A list contains all the training indices.

    Returns
    -------
    np.ndarray
        A 1-d list contains all testing indices.
    """
    testing_indices = np.ones(data_size, dtype=bool)
    testing_indices[training_indices] = False
    return np.array(np.where(testing_indices == True)[0])


def visualize_images(images: np.ndarray, labels: np.ndarray = None,
                     figsize=(10, 10),
                     *args, **kwargs):
    """Visualize 36 random samples from <images> in CelebA dataset.

    Parameters
    ----------
    images : np.ndarray
        Images to display.
    figsize : tuple
        A tuple contains 2 floats that specify width and height in
        inches of the figure.
    labels : np.ndarray
        Corresponding labels of images, 1 dimensional.
    """
    N = images.shape[0]
    plt.figure(figsize=figsize)
    row_num = math.ceil(math.sqrt(N))
    col_num = math.ceil(N/row_num)

    for i in range(N):
        # plt.subplot index starts with 1
        plt.subplot(row_num, col_num, i+1)
        # clear ticks
        plt.xticks([])
        plt.yticks([])
        # remove grids
        plt.grid(False)
        plt.imshow(images[i], *args, **kwargs)
        if labels:
            plt.xlabel(np.squeeze(labels[i], axis=-1))
    plt.show()


def preprocess_batch(sorted_inds, x=None, y=None):
    """Preprocess batched data.

    Parameters
    ----------
    sorted_inds : list
        The sorted indices of the batch.
    x : np.ndarray, optional
        The original input images, by default None.
    y : np.ndarray, optional
        The original labels, by default None.

    Returns
    -------
    np.ndarray, np.ndarray
        The preprocessed batched images and labels.
    """
    if x is not None:
        x = (x[sorted_inds, :, :, ::-1] / 255.).astype(np.float32)
    if y is not None:
        y = y[sorted_inds].astype(np.float32)
    return x, y
