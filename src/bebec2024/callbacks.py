from pathlib import Path
import tensorflow as tf


class TestDatasetCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir=None, name="test"):
        self.dataset = dataset
        self.log_dir = Path(log_dir) / name if log_dir is not None else None
        if self.log_dir is not None:
            self.file_writer = tf.summary.create_file_writer(str(self.log_dir))

    def on_epoch_end(self, epoch, logs=None):
        loss = self.model.evaluate(self.dataset)
        # write loss to tensorboard with tag epoch loss as test loss
        if self.file_writer is not None:
            with self.file_writer.as_default():
                tf.summary.scalar("epoch_loss", loss, step=epoch)
