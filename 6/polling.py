import tensorflow as tf # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras import layers # pyright: ignore[reportMissingModuleSource]

# Shape before adding channel dimension: (batch, height, width).
x = tf.constant([
    [
        [1.0, 3.0, 2.0, 4.0],
        [5.0, 6.0, 1.0, 2.0],
        [7.0, 2.0, 9.0, 1.0],
        [3.0, 4.0, 2.0, 8.0]
    ]
])

# Add one channel: (1, 4, 4, 1).
x = tf.expand_dims(x, axis=-1)

pool = layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="valid"
)

output = pool(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)
print("Pooled values:")
print(output.numpy()[0, :, :, 0])