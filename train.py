import argparse
from pathlib import Path

import ray
import tensorflow as tf

from bebec2024.callbacks import TestDatasetCallback
from bebec2024.config import ConfigBase


def get_config(**kwargs):
    config = ConfigBase.from_toml(kwargs["config"])
    config.to_toml()
    print(100 * "#")
    print(config.sessionname)
    print(100 * "#")
    return config


def get_learning_rate(config):
    if not config.use_lr_scheduler:
        return config.learning_rate
    from bebec2024.utils import LRScheduler

    return LRScheduler(
        learning_rate=config.learning_rate,
        n_epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
        decay_epochs=config.decay_epochs,
        steps_per_epoch=config.steps_per_epoch,
    )


def get_weight_decay(config):
    if not config.use_wd_scheduler:
        return config.weight_decay
    from bebec2024.utils import WDScheduler

    return WDScheduler(
        weight_decay=config.weight_decay,
        n_epochs=config.epochs,
        warmup_epochs=config.warmup_epochs,
        decay_epochs=config.decay_epochs,
        steps_per_epoch=config.steps_per_epoch,
    )


def get_test_callbacks(test_splits, config):
    callbacks = []
    if test_splits:
        for i, tsplit in enumerate(test_splits):
            test_dataset_callback = TestDatasetCallback(
                tsplit, log_dir=config.log_dir, name=f"test_{i}"
            )
            callbacks.append(test_dataset_callback)
    return callbacks


def main(config):
    ray.init(address="local")  # parallel data processing
    config = ConfigBase.from_toml(config)

    # get datasets
    training_splits = [d.get_training_pipeline() for d in config.datasets if d.training]
    assert len(training_splits) == 1, "Only one training dataset is supported"
    validation_splits = [
        d.get_validation_pipeline() for d in config.datasets if d.validation
    ]
    assert len(validation_splits) == 1, "Only one validation dataset is supported"
    test_splits = [d.get_test_pipeline() for d in config.datasets if d.test]

    # build model
    model = config.model.create_instance()

    # compile model
    if config.use_wd_scheduler:
        import tensorflow_addons as tfa

        optimizer = tfa.optimizers.AdamW
    else:
        optimizer = tf.keras.optimizers.AdamW
    model.compile(
        optimizer=optimizer(
            learning_rate=get_learning_rate(config),
            weight_decay=get_weight_decay(config),
        ),
    )

    # set up the callbacks
    callbacks = []

    if config.tensorboard:
        tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=config.log_dir,
            histogram_freq=0,
            write_graph=False,
            write_images=False,
            update_freq="epoch",
        )
        callbacks.append(tb_cb)

    callbacks += get_test_callbacks(test_splits, config)

    if config.ckpt:
        ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                Path(config.ckpt_dir) / "best_ckpt" / "{epoch:04d}-{val_loss:.2f}.keras"
            ),
            monitor="val_loss",
            save_weights_only=False,
            save_best_only=True,
            mode="min",
        )
        callbacks.append(ckpt)

    # train model
    model.fit(
        x=training_splits[0],
        validation_data=validation_splits[0],
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        callbacks=callbacks,
    )

    model.save(str(Path(config.ckpt_dir) / "final_model.keras"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,  #
        help="A .toml file with the training config. See directory /configfiles",
    )
    kwargs = vars(parser.parse_args())
    main(**kwargs)
