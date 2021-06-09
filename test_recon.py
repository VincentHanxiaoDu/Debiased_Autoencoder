from DB_VAE.db_vae import DB_VAE
from Settings.settings import *
from Utils.dataset_utils import *
import h5py

if __name__ == "__main__":
    DATA_FILE_NAME = "CelebA.h5"
    DATA_FILE_PATH = os.path.join(DATASETS_DIR, DATA_FILE_NAME)

    # Download the data file if the file does not exist.
    if not os.path.exists(DATA_FILE_PATH):
        download_CelebA(dir=DATASETS_DIR, filename=DATA_FILE_NAME)

    # Read data.
    original_data = h5py.File(DATA_FILE_PATH, "r")
    original_images = original_data["images"][:]
    original_labels = original_data["labels"][:]
    N = original_labels.shape[0]
    training_indices, testing_indices = generate_training_testing_indices(N)
    
    latent_dim = 100
    db_vae = DB_VAE(latent_dim)
    db_vae.load_weights(CHECKPOINT_PREFIX)
    
    test_logger = logging.getLogger("Test")
    test_logger.setLevel(LOG_LEVEL)
    test_labels = original_labels[testing_indices]

    recon_face_indices = testing_indices[np.where(
        np.squeeze(test_labels, axis=-1) == 1)[0]][0:36]

    x = preprocess_batch(recon_face_indices, x=original_images)[0]
    _, z_mu, z_log_sigma_sq = db_vae.encode(x)
    z = db_vae.sample_latent(z_mu, z_log_sigma_sq)
    reconstruction = np.clip(db_vae.decode(z), 0, 1)
    visualize_images(reconstruction)