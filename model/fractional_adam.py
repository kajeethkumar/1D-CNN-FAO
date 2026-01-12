import tensorflow as tf
from tensorflow.keras.optimizers import Adam

class FractionalAdam(tf.keras.optimizers.Adam):
    """
    Fractional Adam Optimizer (FAO)
    Implements fractional learning-rate scheduling
    without modifying gradient computation.
    """
    def __init__(self,
                 learning_rate=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-7,
                 fraction=0.7,
                 name="FractionalAdam",
                 **kwargs):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            name=name,
            **kwargs
        )
        self.fraction = fraction

    def _decayed_lr(self, var_dtype):
        lr = super()._decayed_lr(var_dtype)
        step = tf.cast(self.iterations + 1, tf.float32)
        t_frac = tf.round(step * self.fraction)
        return lr / (1.0 + t_frac)

    def get_config(self):
        config = super().get_config()
        config.update({"fraction": self.fraction})
        return config
