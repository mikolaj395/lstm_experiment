import tensorflow as tf

from data_generator import DataGenerator

# parameters
params = {
    'sequence_length_min': 45,
    'sequence_length_max': 46,
    't1_min': 10,
    't1_max': 20,
    't2_min': 30,
    't2_max': 40,
    'batch_size': 4,
    'epochs_num': 1,
    'steps_per_epoch': 5000,
}

# Generators
training_generator = DataGenerator(params)

# Design model
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.LSTM(10, batch_input_shape=(params['batch_size'], params['sequence_length_max'], 8),
                              stateful=False))
model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model on dataset
model.fit_generator(generator=training_generator,
                    epochs=params['epochs_num'],
                    steps_per_epoch=params['steps_per_epoch'])
