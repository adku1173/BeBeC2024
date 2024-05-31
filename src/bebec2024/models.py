import tensorflow as tf

from bebec2024.utils import get_pos_encoding_matrix
from bebec2024.loss import distance_matrix, remove_zero_padding, tf_constrained_kmeans


class EigmodePreprocessing(tf.keras.layers.Layer):
    """Implements a preprocessing layer for eigenmode transformer models."""

    def __init__(self, neig=None, concat_real_imag=True):
        super().__init__()
        self.neig = neig
        self.concat_real_imag = concat_real_imag

    def preprocessing(self, csm, neig):
        """Apply the preprocessing to the input tensor.

        :param csm: The input tensor.
        :return: The processed tensor.
        """
        evls, evecs = tf.linalg.eigh(csm)
        return evecs[..., -neig:] * evls[:, tf.newaxis, -neig:]

    def call(self, csm):
        # preprocessing
        neig = 0 if self.neig is None else self.neig
        # get eigmodes
        eigmode = self.preprocessing(csm, neig)
        # concatenate real and imaginary part
        if self.concat_real_imag:
            eigmode = tf.stack([tf.math.real(eigmode), tf.math.imag(eigmode)], axis=3)
            eigmode = tf.transpose(eigmode, [0, 2, 1, 3])
            input_shape = tf.shape(eigmode)
            eigmode = tf.reshape(
                eigmode, [-1, input_shape[1], input_shape[2] * input_shape[3]]
            )
        return eigmode


class MLPViT(tf.keras.layers.Layer):
    """Implements a preprocessing layer for eigenmode transformer models."""

    def __init__(self, hidden_size, dropout_rate, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.fully_connected_layers = [
            tf.keras.layers.Dense(hidden_size, activation=tf.nn.gelu)
            for _ in range(num_layers)
        ]
        self.dropout_layers = [
            tf.keras.layers.Dropout(dropout_rate) for _ in range(num_layers)
        ]

    def call(self, x):
        """Apply the preprocessing to the input tensor.

        :param x: The input tensor.
        :return: The processed tensor.
        """
        for i in range(len(self.fully_connected_layers)):
            x = self.fully_connected_layers[i](x)
            x = self.dropout_layers[i](x)
        return x


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    """Implements a multi-head self-attention layer.

    :param num_heads: The number of attention heads.
    :param key_dim: The size of each attention head.
    :param dropout: The dropout rate to use.
    """

    def __init__(self, num_heads, key_dim, dropout, use_causal_mask=False, version=2):
        super().__init__()
        self.use_causal_mask = use_causal_mask
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout
        )
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add = tf.keras.layers.Add()
        if version == 1:
            self.call_func = self.call_v1
        elif version == 2:
            self.call_func = self.call_v2

    def call_v1(self, x):
        """Apply the multi-head self-attention layer to the input tensor.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.add([self.mha(x, x, use_causal_mask=self.use_causal_mask), x])
        return self.layernorm(x)

    def call_v2(self, x):
        """Apply the multi-head self-attention layer to the input tensor.

        :param x: The input tensor.
        :return: The output tensor.
        """
        y = self.layernorm(x)
        return self.add([self.mha(y, y, use_causal_mask=self.use_causal_mask), x])

    def call(self, x):
        """Apply the multi-head self-attention layer to the input tensor.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.call_func(x)


class TransformerMLP(tf.keras.layers.Layer):
    """Implements a transformer-based multi-layer perceptron layer.

    :param key_dim: The size of the hidden layer in the MLP.
    :param dropout: The dropout rate to use.
    :param version: The version of the transformer MLP to use
                    (1: as in "Attention is all you need" 2: as in: "AN IMAGE IS WORTH 16X16 WORDS" ).
    """

    def __init__(self, key_dim, hidden_dim=None, dropout=0.1, version=2):
        super().__init__()
        if version == 1:
            self.call_func = self.call_v1
            activation = "relu"
        elif version == 2:
            self.call_func = self.call_v2
            activation = tf.nn.gelu
        else:
            raise ValueError("TransformerMLP version must be 1 or 2")

        if hidden_dim is None:
            hidden_dim = key_dim * 2
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim, activation=activation),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(key_dim, activation=activation),
                tf.keras.layers.Dropout(dropout),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call_v1(self, x):
        """Apply the transformer-based MLP layer to the input tensor.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.layer_norm(self.add([x, self.seq(x)]))

    def call_v2(self, x):
        """Apply the transformer-based MLP layer to the input tensor.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.add([x, self.seq(self.layer_norm(x))])

    def call(self, x):
        """Apply the transformer-based MLP layer to the input tensor.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.call_func(x)


class ViTEncoderLayer(tf.keras.layers.Layer):
    """Implements a single encoder layer in a Vision Transformer (ViT) model.

    :param name: The name of the layer.
    :param num_heads: The number of attention heads in the self-attention layer.
    :param key_dim: The size of the hidden layer in the MLP.
    :param dropout: The dropout rate to use.
    """

    def __init__(self, name, num_heads, key_dim, dropout, version=2, hidden_dim=None):
        super().__init__(name=name)
        self.mhsa = MultiHeadSelfAttention(
            num_heads, key_dim, dropout, version=version, use_causal_mask=False
        )
        self.mlp = TransformerMLP(
            key_dim=key_dim, hidden_dim=hidden_dim, dropout=dropout, version=version
        )

    def call(self, x):
        """Apply the ViT encoder layer to the input tensor.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.mlp(self.mhsa(x))


class TransformerEncoder(tf.keras.Model):
    """TransformerEncoder is a stack of `num_layers` Transformer encoder layers.

    :param version: The version of the encoder layer to use (1 or 2). Affects the order of the normalization
    (1: after MHSA (according to Vaswani et al.), 2: before MHSA (ViT)). Default is version 2
    :param num_layers: The number of encoder layers in the model.
    :param key_dim: The dimensionality of the key space (vector length of each vector in the sequence).
    :param num_heads: The number of attention heads in the self-attention layers.
    :param hidden_dim: The size of the hidden layer in the encoder layer MLP (Defaults to None -> 2*key_dim).
    :param dropout_rate: The dropout rate to use.
    """

    def __init__(
        self, version, num_layers, key_dim, num_heads, hidden_dim=None, dropout_rate=0.1
    ):
        super().__init__()
        self.key_dim = key_dim
        self.num_layers = num_layers
        self.enc_layers = [
            ViTEncoderLayer(
                name=f"vit_encoder_layer_{i}",
                num_heads=num_heads,
                key_dim=key_dim,
                dropout=dropout_rate,
                hidden_dim=hidden_dim,
                version=version,
            )
            for i in range(num_layers)
        ]

    def call(self, x):
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x  # Shape `(batch_size, seq_len, key_dim)`.


class EigmodeTransformer(tf.keras.Model):
    """Implements an eigenmode transformer model.

    Refers to the paper: ''Fast grid-free strength mapping of multiple sound sources
    from microphone array data using a Transformer architecture. (2022)''

    Single frequency (band) model using only a transformer encoder. Originally implemented
    for 64 microphones and 10 potential sources.

    :param batchsize: The batchsize used for training
    :param nsources: The number of sources in the output tensor.
    :param dim: The dimensionality of the source locations.
    :param noise_power: A flag indicating whether the noise power is included in the strength vector.
    :param num_layers: The number of encoder layers in the model.
    :param num_heads: The number of attention heads in the self-attention layers.
    :param projection_dim: The size of the hidden layer in the MLP.
    :param dropout_rate: The dropout rate to use.
    :param lam: The weighting between location and alpha loss
    """

    def __init__(
        self,
        batchsize,
        encoder=None,
        neig=None,
        preprocess=True,
        nsources=10,
        nchannels=64,
        dim=2,
        noise_power=False,
        pos_enc=False,
        freq_enc=False,
        max_seq_len=64,
        freq_enc_pos="mlp",
        nfreqs=None,
        dropout_rate=0.1,
        hidden_layers=1,
        lam=5.0,
    ):
        super(EigmodeTransformer, self).__init__()
        self.batchsize = batchsize
        self.max_seq_len = max_seq_len
        if neig is not None:
            self.max_seq_len = neig
        self.pos_enc = pos_enc
        self.freq_enc = freq_enc
        self.freq_enc_pos = freq_enc_pos
        self.neig = neig
        self.preprocess = preprocess
        self.noise_power = noise_power
        self.nsources = nsources
        self.nchannels = nchannels
        self.lam = lam
        self.dim = dim
        self.dropout_rate = dropout_rate
        # layers before head is applied
        self.preprocessing = EigmodePreprocessing(neig)
        self.encoder = (
            TransformerEncoder(
                version=2,
                num_layers=12,
                key_dim=self.nchannels * 2,
                num_heads=12,
                dropout_rate=dropout_rate,
            )
            if encoder is None
            else encoder
        )
        # positional embedding
        if self.pos_enc:
            self.positional_encoding = tf.keras.layers.Embedding(
                input_dim=self.max_seq_len,
                output_dim=self.encoder.key_dim,
                weights=[
                    get_pos_encoding_matrix(self.max_seq_len, self.encoder.key_dim)
                ],
                trainable=False,
                name="position_embedding",
            )
        self.flatten = tf.keras.layers.Flatten()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout_head = tf.keras.layers.Dropout(dropout_rate)
        self.strength_head = tf.keras.layers.Dense(units=nsources, activation="linear")
        self.loc_head = [
            tf.keras.layers.Dense(units=nsources, activation="linear")
            for _ in range(dim)
        ]
        if noise_power:
            self.noise_head = tf.keras.layers.Dense(units=1, activation="linear")
        self.dense_head = MLPViT(
            dropout_rate=dropout_rate,
            hidden_size=self.encoder.key_dim * 4,
            num_layers=hidden_layers,
        )

        # frequency embedding
        if self.freq_enc:
            if self.freq_enc_pos not in ["mlp"]:
                raise ValueError("freq_enc_pos must be mlp")
        # metrics
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        """Apply the EigenmodeTransformer to the input tensor.

        :param x: The input tensor.
        :return: The output tensor (individual source strength, source positions, sensor noise power).
        """
        if self.freq_enc:
            # assert len(inputs) == 2, 'frequency encoding is not provided in inputs'
            inputs, freq = inputs
            if freq.ndim == 1:
                freq = freq[:, tf.newaxis]

        # preprocessing
        if self.preprocess:  # csm to eigenmodes
            inputs = self.preprocessing(inputs)
        if self.pos_enc:
            encodings = self.positional_encoding(
                tf.range(start=0, limit=self.max_seq_len, delta=1)
            )
            inputs += encodings

        # transformer encoder
        x = self.encoder(inputs)
        # mlp head
        x = self.norm(x)
        x = self.flatten(x)
        if self.freq_enc and self.freq_enc_pos == "mlp":
            x = tf.concat([x, freq], axis=-1)
        x = self.dropout_head(x)
        x = self.dense_head(x)
        loc = tf.concat(
            [self.loc_head[i](x)[:, tf.newaxis, :] for i in range(self.dim)], axis=1
        )
        if self.noise_power:
            p2 = tf.nn.softmax(
                tf.concat([self.strength_head(x), self.noise_head(x)], axis=-1)
            )
        else:
            p2 = tf.nn.softmax(self.strength_head(x))
        return (p2[:, : self.nsources], loc, p2[:, self.nsources :])

    @tf.function()
    def get_assignment(self, loc_true, loc_pred):
        def loop_body(x):
            i, ltrue, lpred = x
            _loc_true = remove_zero_padding(ltrue)
            Rho = distance_matrix(_loc_true, lpred)
            I = tf.shape(Rho)[0]  # I: no. of estimates ; J: no. of true sources
            J = tf.shape(Rho)[1]
            tau = tf.cast(
                tf.math.floordiv(I, J), tf.int16
            )  # blanced or near-balanced assignment
            T = tf_constrained_kmeans(Rho, tau)
            # Padding with zeros
            T = tf.pad(T, [[0, 0], [0, I - J]], "CONSTANT", constant_values=0.0)
            Rho = tf.pad(Rho, [[0, 0], [0, I - J]], "CONSTANT", constant_values=0.0)
            return T, Rho

        T, Rho = tf.map_fn(
            loop_body,
            elems=(tf.range(self.batchsize), loc_true, loc_pred),
            fn_output_signature=(
                tf.TensorSpec(shape=(self.nsources, self.nsources), dtype=tf.float32),
                tf.TensorSpec(shape=(self.nsources, self.nsources), dtype=tf.float32),
            ),
            parallel_iterations=self.batchsize,
        )
        return tf.abs(T), Rho

    @tf.function(jit_compile=True)
    def matching_loss(self, T, Rho, alpha_true, alpha_pred):
        # alpha loss
        estimate_sum = tf.einsum("bij,bi->bj", T, alpha_pred[:, : self.nsources])
        loss = tf.reduce_sum(
            (alpha_true[:, : self.nsources] - estimate_sum) ** 2, axis=1
        )
        # noise loss
        loss += tf.reduce_sum(
            (alpha_true[:, self.nsources :] - alpha_pred[:, self.nsources :]) ** 2,
            axis=1,
        )
        loss *= self.lam
        # loc loss
        loss += tf.reduce_sum(Rho * T, axis=[1, 2])
        return tf.reduce_mean(loss)

    @tf.function()
    def get_alpha_and_loc(self, labels):
        if self.noise_power:
            assert len(labels) == 3, "noise power is not provided in labels"
            strength_true, loc_true, noise_true = labels
            if tf.shape(tf.shape(noise_true))[0] == 1:
                noise_true = noise_true[:, tf.newaxis]
            alpha_true = tf.concat([strength_true, noise_true], axis=1)
            return alpha_true, loc_true
        else:
            return labels[0], labels[1]

    @tf.function()
    def train_step(self, data):
        csm, labels = data
        alpha_true, loc_true = self.get_alpha_and_loc(labels)
        # start training
        with tf.GradientTape() as tape:
            (alpha_pred, loc_pred, noise_pred) = self(
                csm, training=True
            )  # Forward pass
            T, Rho = self.get_assignment(
                loc_true, loc_pred
            )  # abs in case of rounding errors
            if self.noise_power:
                alpha_pred = tf.concat([alpha_pred, noise_pred], axis=1)
            loss = self.matching_loss(T, Rho, alpha_true, alpha_pred)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        if hasattr(self, "compiled_metrics") and self.compiled_metrics is not None:
            self.compiled_metrics.update_state(
                (alpha_true, loc_true), (alpha_pred, loc_pred)
            )
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function()
    def test_step(self, data):
        csm, labels = data
        alpha_true, loc_true = self.get_alpha_and_loc(labels)
        y_pred = self(csm, training=False)
        (alpha_pred, loc_pred, noise_pred) = y_pred  # Forward pass
        T, Rho = self.get_assignment(
            loc_true, loc_pred
        )  # abs in case of rounding errors
        if self.noise_power:
            alpha_pred = tf.concat([alpha_pred, noise_pred], axis=1)
        loss = self.matching_loss(T, Rho, alpha_true, alpha_pred)
        # Update the metrics.
        self.loss_tracker.update_state(loss)
        if hasattr(self, "compiled_metrics") and self.compiled_metrics is not None:
            self.compiled_metrics.update_state(
                (alpha_true, loc_true), (alpha_pred, loc_pred)
            )
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
