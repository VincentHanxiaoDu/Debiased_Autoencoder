from Settings.settings import *
from Utils.dataset_utils import *
from DB_VAE.db_vae import DB_VAE
import os
import h5py


def get_batch_indices(face_images_indices, non_face_images_indices,
                      face_batch_size, non_face_batch_size, p_faces):
    # sample face images according to p_faces
    face_indices = np.random.choice(face_images_indices,
                                    size=face_batch_size,
                                    replace=False, p=p_faces)

    # uniformly sample non-face images.
    non_face_indices = np.random.choice(non_face_images_indices,
                                        size=non_face_batch_size, replace=False)
    batch_inds = np.concatenate((face_indices, non_face_indices))
    return np.sort(batch_inds)


def train_model(db_vae: DB_VAE, original_images, original_labels,
                training_indices, optimizer, batch_size=32, epochs=6):
    # Logger.
    train_logger = logging.getLogger("Train")
    train_logger.setLevel(LOG_LEVEL)

    # clear tqdm instances.
    if hasattr(tqdm, '_instances'):
        tqdm._instances.clear()

    # Get training indices of faces and non-faces.
    training_labels = original_labels[training_indices]
    training_face_images_indices = training_indices[np.where(
        np.squeeze(training_labels, axis=-1) == 1)[0]]
    training_non_face_images_indices = training_indices[np.where(
        np.squeeze(training_labels, axis=-1) == 0)[0]]

    # Define numbers of face and non-face images in batch.
    face_batch_size = batch_size//2
    non_face_batch_size = batch_size - face_batch_size

    loss_history = []
    for epoch in range(epochs):
        train_logger.info(f"Starting epoch {epoch+1}/{epochs}")
        train_logger.info(f"Getting training sample probabilities.")
        p_faces = db_vae.get_training_sample_probabilities(
            original_images, training_face_images_indices)
        for _ in tqdm(range(N // batch_size)):
            # Get next batch.
            batch_image_indices = get_batch_indices(
                training_face_images_indices, training_non_face_images_indices,
                face_batch_size, non_face_batch_size, p_faces)
            batch_x, batch_y = preprocess_batch(
                batch_image_indices, original_images, original_labels)

            # train step
            loss = db_vae.train_step(batch_x, batch_y, optimizer)
        db_vae.save_weights(CHECKPOINT_PREFIX)
        train_logger.info(f"Model loss of current epoch: {loss}")
    train_logger.info("Model trained successfully!")
    return loss_history


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

    # Get training and testing dataset infices.
    N = original_labels.shape[0]
    training_indices, testing_indices = generate_training_testing_indices(N)

    # Construct model.
    batch_size = 32
    epochs = 6
    learning_rate = 5e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    latent_dim = 100
    db_vae = DB_VAE(latent_dim)

    train_model(db_vae, original_images, original_labels, training_indices,
                optimizer)

    test_logger = logging.getLogger("Test")
    test_logger.setLevel(LOG_LEVEL)
    y_pred = tf.round(tf.sigmoid(
        db_vae.predict(original_images, testing_indices)))
    test_labels = original_labels[testing_indices]
    test_accuracy = tf.equal(y_pred, test_labels).numpy().mean()
    test_logger.info(f"Test accuracy: {test_accuracy}")

    
    
    recon_face_indices = testing_indices[np.where(
        np.squeeze(test_labels, axis=-1) == 1)[0]][0:36]

    _, z_mu, z_log_sigma_sq = db_vae.encode(original_images[recon_face_indices])
    
    z = db_vae.sample_latent(z_mu, z_log_sigma_sq)
    reconstruction = np.clip(db_vae.decode(z), 0, 1)
