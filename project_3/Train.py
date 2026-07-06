import os
import tensorflow as tf

# Shortcuts so in case of form tensforflow.keras import layers not work
layers = tf.keras.layers
models = tf.keras.models

#  GPU setup : Check if TensorFlow can actually see a GPU. 
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

if gpus:
    try:
        # By default TF grabs ALL GPU memory upfront. Memory growth makes it allocate only what it needs, 
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled.")
    except RuntimeError as e:
        # This has to be set before GPUs are initialized, so if something already touched the GPU context, this will fail. Not fatal, just log it.
        print("Could not set memory growth:", e)
else:
    print("WARNING: No GPU detected - training will run on CPU.")

#  Config for the future use
DATA_DIR = "project_3/PetImages/"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
VAL_SPLIT = 0.2
SEED = 42  # fixed seed so train/val split is reproducible across runs
MODEL_PATH = "project_3/cat_dog_model.h5"

# image_dataset_from_directory expects subfolders per class, e.g. PetImages/Cat/*.jpg and PetImages/Dog/*.jpg. It infers labels from folder names.
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="training",
    seed=SEED,  
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",  
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=VAL_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

class_names = train_ds.class_names  # e.g. ['Cat', 'Dog']
print("Classes:", class_names)

# The PetImages dataset is notorious for having a handful of corrupted/ truncated JPEGs. Rather than manually filtering them out, just skip any file that throws an error during decoding.
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

#  Data augmentation Only applied to training data. Helps the model generalize instead of memorizing exact pixel arrangements (flip, small rotation, small zoom).
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

def preprocess(ds, augment=False):
    if augment:
        # Doing augmentation on CPU frees up the GPU to focus on the actual training compute, and avoids some GPU op compatibility issues.
        with tf.device('/CPU:0'):
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

    # Rescale pixel values from [0, 255] to [0, 1] - helps the network train faster/stabler.
    ds = ds.map(lambda x, y: (x / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)

    # cache() keeps decoded images in memory after the first epoch (speeds up later epochs), prefetch() overlaps data loading with model execution so the GPU doesn't sit idle.
    return ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = preprocess(train_ds, augment=True)
val_ds = preprocess(val_ds, augment=False)  # no augmentation for validation - we want real data

# Pretty standard CNN: 4 conv+pool blocks that progressively extract more abstract features while shrinking spatial size, then a dense head for the final classification.
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),  # 150x150 RGB images

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dropout(0.5),  # randomly drop half the units during training to reduce overfitting
    layers.Dense(256, activation="relu"),
    layers.Dense(1, activation="sigmoid"),  # single output, squashed to [0,1] -> prob of "dog"
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",  # standard loss for binary classification
    metrics=["accuracy"],
)

model.summary()  # quick sanity check on layer shapes and param counts

#  Callbacks 
callbacks = [
    # Stop training if val loss doesn't improve for 4 epochs in a row, and roll back to the best-performing weights instead of the final ones.
    tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),

    # If val loss plateaus for 2 epochs, cut the learning rate in half # helps the model fine-tune once it's close to a good solution.
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
]

#  Train on data
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

#  Evaluation of the model
val_loss, val_acc = model.evaluate(val_ds)
print(f"\nValidation accuracy: {val_acc:.4f}")
print(f"Validation loss: {val_loss:.4f}")

#  Save everything needed to reuse the model later 
model.save(MODEL_PATH)

# Save class names too, since at inference time we'll need to map   model's 0/1 output back to "Cat"/"Dog" and the folder order isn't guaranteed.
with open("project_3/class_names.txt", "w") as f:
    f.write("\n".join(class_names))

print(f"\nModel saved to {MODEL_PATH}")
print(f"Class names saved to class_names.txt -> {class_names}")