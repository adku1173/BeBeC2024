import numpy as np
import tensorflow as tf


def set_memory_growth():
    gpu_available = tf.config.list_physical_devices("GPU")
    if gpu_available:
        print(f"available GPUs: {gpu_available}")
    # only occupy needed GPU RAM
    for gpu in gpu_available:
        tf.config.experimental.set_memory_growth(gpu, True)


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def tf_L_p(x):  # noqa: N802
    p0 = 4e-10
    arg = tf.div(x, p0)
    arg_clip = tf.clip_by_value(
        arg, 1e-35, 10**20
    )  # maximum is unrealistic because none type is not allowed
    return 10 * log10(arg_clip)


def get_pos_encoding_matrix(max_seq_len, key_dim):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / key_dim) for j in range(key_dim)]
            if pos != 0
            else np.zeros(key_dim)
            for pos in range(max_seq_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
    return pos_enc


def he_to_freq(dataset, he_min, he_max):
    fftfreq = dataset.config.freq_data.fftfreq()
    ap = dataset.config.mics.aperture
    c = dataset.config.env.c
    if he_min == he_max:
        f = he_min * c / ap
        ind = np.searchsorted(fftfreq, f)
        return [fftfreq[ind]]
    f_min = he_min * c / ap
    f_max = he_max * c / ap
    ind_low = np.searchsorted(fftfreq, f_min)
    ind_high = np.searchsorted(fftfreq, f_max)
    return list(fftfreq[ind_low : ind_high + 1])


def freq_to_he(dataset, f):
    ap = dataset.config.mics.aperture
    c = dataset.config.env.c
    return [ap * fftc / c for fftc in f]


class Scheduler(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, nominal_value, n_epochs, warmup_epochs, decay_epochs, steps_per_epoch
    ):
        super().__init__()
        self.nominal_value = tf.cast(nominal_value, tf.float64)

        self.steps_per_epoch = steps_per_epoch
        self.decay_epochs = decay_epochs
        self.n_epochs = n_epochs

        self.total_steps = n_epochs * steps_per_epoch
        self.warmup_steps = tf.cast(warmup_epochs * steps_per_epoch, tf.float64)
        self.start_decay_step = tf.cast(
            (n_epochs - decay_epochs) * steps_per_epoch, tf.int64
        )

    def initial_value(self):
        return tf.cast(self.nominal_value / self.warmup_steps * 0.5, tf.float64)

    def linear_ramp(self, step):
        step = tf.cast(step, tf.float64)
        return tf.cast(
            tf.minimum(
                self.nominal_value / self.warmup_steps * step, self.nominal_value
            ),
            tf.float64,
        )

    def exponential_decay(self, step):
        factor = tf.cast(step - self.start_decay_step, tf.float64)
        return (
            tf.exp(tf.cast(-0.05 / self.steps_per_epoch, tf.float64) * factor)
            * self.nominal_value
        )

    def __call__(self, step):
        step = tf.cast(step, tf.int64)
        val = tf.cond(
            step == tf.cast(0, tf.int64),
            lambda: self.initial_value(),
            lambda: self.linear_ramp(step),
        )
        val = tf.cond(
            step < self.start_decay_step,
            lambda: val,
            lambda: self.exponential_decay(step),
        )
        return val

    def get_config(self):
        config = {
            "nominal_value": float(self.nominal_value),
            "warmup_epochs": float(self.warmup_steps),
            "n_epochs": int(self.n_epochs),
            "steps_per_epoch": int(self.steps_per_epoch),
            "decay_epochs": int(self.decay_epochs),
        }
        return config


class LRScheduler(Scheduler):
    def __init__(
        self,
        learning_rate,
        n_epochs,
        warmup_epochs,
        decay_epochs,
        steps_per_epoch: int = 2000,
    ):
        super().__init__(
            nominal_value=learning_rate,
            n_epochs=n_epochs,
            warmup_epochs=warmup_epochs,
            decay_epochs=decay_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        self.nominal_learning_rate = learning_rate

    def get_config(self):
        config = super().get_config()
        config.pop("nominal_value")
        config["learning_rate"] = float(self.nominal_learning_rate)
        return config


class WDScheduler(Scheduler):
    def __init__(
        self,
        weight_decay,
        n_epochs,
        warmup_epochs,
        decay_epochs,
        steps_per_epoch: int = 2000,
    ):
        super().__init__(
            nominal_value=weight_decay,
            n_epochs=n_epochs,
            warmup_epochs=warmup_epochs,
            decay_epochs=decay_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        self.nominal_weight_decay = weight_decay

    def linear_ramp(self, step):
        step = tf.cast(step, tf.float64)
        value = self.nominal_value * ((step / self.warmup_steps) * 0.5 + 0.5)
        return tf.cast(tf.minimum(value, self.nominal_value), tf.float64)

    def __call__(self, step):
        val = tf.cond(
            step < self.start_decay_step,
            lambda: self.linear_ramp(step),
            lambda: self.exponential_decay(step),
        )
        return val

    def get_config(self):
        config = super().get_config()
        config.pop("nominal_value")
        config["weight_decay"] = float(self.nominal_weight_decay)
        return config
