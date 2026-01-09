import logging
from tensorflow.keras import Model, backend as K
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam


class VAE:
    def __init__(self, original_dim, intermediate_dim, latent_dim):
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        # Create the full VAE model by connecting encoder outputs directly to decoder
        self.model = Model(
            inputs=self.encoder.input,
            outputs=self.decoder(self.encoder.get_layer("z").output),
            name="vae_mlp",
        )

    def sample(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def build_encoder(self):
        inputs = Input(shape=(self.original_dim,), name="encoder_input")
        x = Dense(self.intermediate_dim, activation="relu")(inputs)
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)
        z = Lambda(self.sample, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var])
        return Model(inputs, z, name="encoder")  # Return only z

    def build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,), name="z_sampling")
        x = Dense(self.intermediate_dim, activation="relu")(latent_inputs)
        outputs = Dense(self.original_dim, activation="sigmoid")(x)
        return Model(latent_inputs, outputs, name="decoder")

    def compile(self, optimizer="adam", learning_rate=0.0001, clipvalue=0.5):
        opt = Adam(learning_rate=learning_rate, clipvalue=clipvalue)
        self.model.compile(optimizer=opt, loss=self.vae_loss)

    def vae_loss(self, x, x_decoded_mean):
        z_mean, z_log_var = (
            self.encoder(x)[1],
            self.encoder(x)[2],
        )  # assuming these are the second and third outputs
        reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return total_loss

    def train(self, x_train, batch_size=32, epochs=10):
        self.model.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_train, x_train),
        )