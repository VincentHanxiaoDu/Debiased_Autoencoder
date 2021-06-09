import os
import tensorflow as tf
from pathlib import Path
import logging

# Paths
SETTINGS_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_ROOT_DIR = SETTINGS_DIR.parent.absolute()
DATASETS_DIR = os.path.join(PROJECT_ROOT_DIR, "Datasets")

# Logging
LOG_FORMAT = "[%(levelname)s] %(asctime)s %(name)s: %(message)s"
LOG_LEVEL = logging.INFO
logging.basicConfig(format=LOG_FORMAT)

# Model
MODEL_NAME = "db_vae"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT_DIR, "training_checkpoints")
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, MODEL_NAME + "_ckpt")
