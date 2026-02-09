import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import sys

# --- Data Preparation ---
# Load CIFAR-10 instead of MNIST
(train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10 images are 32x32x3. Normalize to [-1, 1]
train_images = train_images.astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 50000
BATCH_SIZE = 64
steps_per_epoch = len(train_images) // BATCH_SIZE

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# --- Model Definitions ---

def make_generator_model():
    model = tf.keras.Sequential([
        # Start with 8x8 image to eventually reach 32x32 (8*2*2 = 32)
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),

        # 8x8 -> 16x16
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # 16x16 -> 32x32
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Output: 32x32x3 (The 3 represents RGB channels)
        layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
    ])
    return model


def make_discriminator_model():
    model = tf.keras.Sequential([
        # Input shape updated for 32x32x3
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[32, 32, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


generator = make_generator_model()
discriminator = make_discriminator_model()

# --- Loss & Optimizers ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Static seed for consistent progress tracking
seed = tf.random.normal([16, 100])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    grad_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grad_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(grad_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(grad_disc, discriminator.trainable_variables))
    return gen_loss, disc_loss


def generate_and_save_images(model, epoch, test_input):
    # Notice: training=False is important for BatchNormalization
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # Rescale from [-1, 1] to [0, 1] for RGB display
        img_to_show = (predictions[i] * 127.5 + 127.5) / 255.0
        plt.imshow(img_to_show)
        plt.axis('off')

    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close()


def train(dataset, epochs):
    print(f"Starting Training for {epochs} Epochs...")
    history = []

    for epoch in range(epochs):
        start = time.time()
        batch_idx = 0
        last_gen_loss = 0
        last_disc_loss = 0

        for image_batch in dataset:
            batch_idx += 1
            gen_loss, disc_loss = train_step(image_batch)
            last_gen_loss = gen_loss
            last_disc_loss = disc_loss

            percent = (batch_idx / steps_per_epoch) * 100
            print(
                f"Epoch {epoch + 1} Progress: {percent:>6.2f}% | Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss:.4f}",
                end='\r')

        epoch_duration = time.time() - start

        # Store metrics for MD table
        history.append({
            "epoch": epoch + 1,
            "gen_loss": last_gen_loss.numpy(),
            "disc_loss": last_disc_loss.numpy(),
            "time": epoch_duration
        })

        # Save images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            print(f"\n[Checkpoint] Saved images for Epoch {epoch + 1}")

        print(f"\nEpoch {epoch + 1} Finished in {epoch_duration:.2f}s\n" + "-" * 50)

    # --- Generate MD Table Output ---
    print("\n### Training Summary (Markdown Format)\n")
    md_table = "| Epoch Number | Generator Loss | Discriminator Loss | Training Time (s) |\n"
    md_table += "| :--- | :--- | :--- | :--- |\n"
    for log in history:
        md_table += f"| {log['epoch']} | {log['gen_loss']:.4f} | {log['disc_loss']:.4f} | {log['time']:.2f} |\n"
    print(md_table)


# Run the training
train(train_dataset, epochs=50)