def CNN() : 
    # Create the model
    cnn_model = models.Sequential()
    cnn_model.add(layers.Conv2D(64, (5, 5), activation='relu', input_shape=(400, 400, 3), padding='same', use_bias=True))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(128, (5, 5), activation='relu', input_shape=(8, 8, 3), padding='same', use_bias=True))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(128, activation='relu'))
    cnn_model.add(layers.Dense(2))
    return cnn_model