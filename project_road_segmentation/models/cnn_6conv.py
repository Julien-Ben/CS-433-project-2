from tensorflow.keras.layers import Sequential, Conv2D, MaxPooling2D, LeakyReLU, Dropout, Flatten , Dense
from tensorflow.keras.regularizers import l2

def cnn_6conv():
    input_shape = (16,16)
    # Size of pooling area for max pooling
    pool_size = (2, 2)

    reg = 1e-6 # L2 regularization factor (used on weights, but not biases)

    model = Sequential()

    model.add(Conv2D(64, (5, 5), # 64 5x5 filters
                            padding='same',
                            input_shape=input_shape,
                            #activation='relu'
                    ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), # 128 3x3 filters
                            padding='same',
                            #activation='relu'
                               ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                            ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, (3, 3), # 256 3x3 filters
                            padding='same',
                            #activation='relu'
                           ))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128,
                    kernel_regularizer=l2(reg),
                    #activation='relu'
                        )) # Fully connected layer (128 neurons)
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))

    model.add(Dense(2,
                    kernel_regularizer=l2(reg),
                    activation='softmax'
                        ))

    return model
