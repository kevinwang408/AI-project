from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN, tcn_full_summary

def build_tcn_model(input_shape):
    """Build a TCN regression model using keras-tcn."""
    tcn_layer = TCN(
        nb_filters=110,
        kernel_size=10,
        dropout_rate=0.4331,
        activation='relu',
        padding='causal',
        nb_stacks=1,
        dilations=(1, 2, 4, 8, 16, 32),
        input_shape=input_shape
    )
    print('Receptive field size =', tcn_layer.receptive_field)
    model = Sequential([
        tcn_layer,
        Dense(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    tcn_full_summary(model, expand_residual_blocks=False)
    return model