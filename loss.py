import numpy as np
import tensorflow as tf
from keras import backend as K


def build_masked_loss(loss_function, mask_value=0):
    """Builds a loss function that masks based on targets

    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets

    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        mask = K.cast(
            K.not_equal(K.cast(K.argmax(y_true, axis=3), dtype="float32"), mask_value),
            K.floatx(),
        )
        # mask = K.reshape(mask,(2,y_pred.shape[1],y_pred.shape[2],1))
        # mask = K.repeat_elements(mask,y_pred.shape[-1],axis=3)
        print(
            tf.boolean_mask(y_true, K.cast(mask, dtype="int32")),
            tf.boolean_mask(y_pred, K.cast(mask, dtype="int32")),
        )
        # print(tf.boolean_mask(y_pred, K.cast(mask, dtype='int32')))
        # return loss_function(y_true * mask, y_pred * mask)
        return loss_function(
            tf.boolean_mask(y_true, K.cast(mask, dtype="int32")),
            tf.boolean_mask(y_pred, K.cast(mask, dtype="int32")),
        )

    return masked_loss_function


def masked_categorical_crossentropy(
    y_true, y_pred, mask_value=np.array([[1.0, 0.0, 0.0]])
):
    # find out which timesteps in `y_true` are not the padding character '#'
    mask = K.all(K.equal(y_true, mask_value), axis=-1)
    mask = 1 - K.cast(mask, K.floatx())

    # multiply categorical_crossentropy with the mask
    loss = K.categorical_crossentropy(y_true, y_pred) * mask

    # take average w.r.t. the number of unmasked entries
    return K.sum(loss) / K.sum(mask)


def get_loss(mask_value):
    mask_value = K.variable(mask_value)

    return masked_categorical_crossentropy


def masked_accuracy(y_true, y_pred, mask_value=0):
    y_true = K.cast(K.argmax(y_true, axis=3), dtype="float32")
    y_pred = K.cast(K.argmax(y_pred, axis=3), dtype="float32")
    mask = K.cast(K.not_equal(y_true, mask_value), dtype="float32")
    total = K.sum(K.cast(K.not_equal(y_true, mask_value), dtype="float32"))
    # correct = K.sum(K.cast(K.equal(y_true*mask, y_pred*mask), dtype='float32')) - K.sum(K.cast(K.equal(y_true, mask_value), dtype='float32'))

    correct = K.sum(
        K.cast(
            K.equal(
                tf.boolean_mask(y_true, K.cast(mask, dtype="int32")),
                tf.boolean_mask(y_pred, K.cast(mask, dtype="int32")),
            ),
            dtype="float32",
        )
    )
    return correct / total


# new losses    -----------------------

# https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def categorical_focal_loss(alpha, gamma=2.0):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(K.sum(loss, axis=-1))

    return categorical_focal_loss_fixed


def dice_loss(y_true, y_pred, smooth=1e-3):
    axis_reduce = (1, 2)
    intersection = K.sum(K.abs(y_true * y_pred), axis=axis_reduce)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=axis_reduce)
    dice = 2 * (intersection + smooth) / (sum_ + smooth)
    return 1 - K.mean(dice)


def jaccard_loss(y_true, y_pred, smooth=1e-3):
    axis_reduce = (1, 2)
    intersection = K.sum(K.abs(y_true * y_pred), axis=axis_reduce)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=axis_reduce)
    jaccard = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1 - K.mean(jaccard)
