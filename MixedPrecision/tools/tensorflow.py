import tensorflow as tf


def float32_variable_storage_getter(getter, name, shape=None, dtype=None,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    """
        Custom variable getter that forces trainable variables to be stored in
        float32 precision and then casts them to the training precision.
    """
    storage_dtype = tf.float32 if trainable else dtype

    variable = getter(
        name,
        shape,
        dtype=storage_dtype,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        *args,
        **kwargs
    )

    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)

    return variable


def gradients_with_loss_scaling(loss, variables, loss_scale):
    """
        Gradient calculation with loss scaling to improve numerical stability when training with float16.
    """
    return [grad / loss_scale for grad in tf.gradients(loss * loss_scale, variables)]
