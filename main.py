import tensorflow as tf

from data_generator import DataGenerator

# parameters
params = {
    'sequence_length': 20,
    't1': 3,
    't2': 6,
    'batch_size': 4,
    'epochs_num': 3,
    'steps_per_epoch': 500,
}

# Generators
training_generator = DataGenerator(params)
validation_generator = DataGenerator(params)

# Design model
model = tf.keras.Sequential()
model.add(
    tf.keras.layers.LSTM(50, batch_input_shape=(params['batch_size'], params['sequence_length'], 8), stateful=True))
model.add(tf.keras.layers.Dense(4, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model on dataset
model.fit_generator(generator=training_generator,
                    epochs=params['epochs_num'],
                    steps_per_epoch=params['steps_per_epoch'],
                    validation_data=validation_generator,
                    validation_steps=params['steps_per_epoch'])
