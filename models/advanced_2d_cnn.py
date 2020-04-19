import numpy as np
import random
import math
import scipy
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

###########################
# Custom Keras layers
###########################

class OneToTwo(tf.keras.layers.Layer):
    ''' Transform 1d to 2d with i,j vectors operated on.'''
    def __init__(self, operation='mean'):
        super(OneToTwo, self).__init__()

    def call(self, oned):
        _, seq_len, features = oned.shape

        twod1 = tf.tile(oned, [1, seq_len, 1])
        twod1 = tf.reshape(twod1, [-1, seq_len, seq_len, features])
        twod2 = tf.transpose(twod1, [0,2,1,3])

        twod1 = tf.expand_dims(twod1, axis=-1)
        twod2 = tf.expand_dims(twod2, axis=-1)
        twod  = tf.concat([twod1, twod2], axis=-1)
        twod = tf.reduce_mean(twod, axis=-1)

        return twod

    def get_config(self):
        config = super().get_config().copy()
        config['operation'] = self.operation
        return config


class ConcatDist2D(tf.keras.layers.Layer):
    ''' Concatenate the pairwise distance to 2d feature matrix.'''
    def __init__(self):
        super(ConcatDist2D, self).__init__()

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_len = input_shape[0], input_shape[1]

        ## concat 2D distance ##
        pos = tf.expand_dims(tf.range(0, seq_len), axis=-1)
        matrix_repr1 = tf.tile(pos, [1, seq_len])
        matrix_repr2 = tf.transpose(matrix_repr1, [1, 0])
        dist = tf.math.abs(tf.math.subtract(matrix_repr1, matrix_repr2))
        dist = tf.dtypes.cast(dist, tf.float32)
        dist = tf.expand_dims(dist, axis=-1)
        dist = tf.expand_dims(dist, axis=0)
        dist = tf.tile(dist, [batch_size, 1, 1, 1])
        return tf.concat([inputs, dist], axis=-1)


class Symmetrize2D(tf.keras.layers.Layer):
    '''Take the average of a matrix and its transpose to enforce symmetry.'''

    def __init__(self):
        super(Symmetrize2D, self).__init__()

    def call(self, x):
        x_t = tf.transpose(x, [0, 2, 1, 3])
        x_sym = (x + x_t) / 2
        return x_sym


class UpperTri(tf.keras.layers.Layer):
    ''' Unroll matrix to its upper triangular portion.'''

    def __init__(self, diagonal_offset=2):
        super(UpperTri, self).__init__()
        self.diagonal_offset = diagonal_offset

    def call(self, inputs):
        seq_len = inputs.shape[1].value
        output_dim = inputs.shape[-1]

        triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
        triu_index = list(triu_tup[0] + seq_len * triu_tup[1])
        unroll_repr = tf.reshape(inputs, [-1, seq_len ** 2, output_dim])
        return tf.gather(unroll_repr, triu_index, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config['diagonal_offset'] = self.diagonal_offset
        return config


class StochasticReverseComplement(tf.keras.layers.Layer):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self):
        super(StochasticReverseComplement, self).__init__()

    def call(self, seq_1hot, training=None):
        if training:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            reverse_bool = tf.random.uniform(shape=[]) > 0.5
            src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, tf.constant(False)


class StochasticShift(tf.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, pad='uniform'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.augment_shifts = tf.range(-self.shift_max, self.shift_max + 1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,
                                        maxval=len(self.augment_shifts))
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(tf.not_equal(shift, 0),
                                lambda: shift_sequence(seq_1hot, shift),
                                lambda: seq_1hot)
            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shift_max': self.shift_max,
            'pad': self.pad
        })
        return config


def shift_sequence(seq, shift, pad_value=0.25):
    """Shift a sequence left or right by shift_amount.
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError('input sequence should be rank 3')
    input_shape = seq.shape

    pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(tf.greater(shift, 0),
                   lambda: _shift_right(seq),
                   lambda: _shift_left(seq))
    sseq.set_shape(input_shape)

    return sseq


class SwitchReverse(tf.keras.layers.Layer):
    """Reverse predictions if the inputs were reverse complemented."""

    def __init__(self):
        super(SwitchReverse, self).__init__()

    def call(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        xd = len(x.shape)
        if xd == 3:
            rev_axes = [1]
        elif xd == 4:
            rev_axes = [1, 2]
        else:
            raise ValueError('Cannot recognize SwitchReverse input dimensions %d.' % xd)

        return tf.keras.backend.switch(reverse,
                                       tf.reverse(x, axis=rev_axes),
                                       x)


###########################
# Helper functions
###########################

def activate(current, activation, verbose=False):
    if verbose:
        print('activate:',activation)

    if activation == 'relu':
        current = tf.keras.layers.ReLU()(current)
    elif activation == 'gelu':
        current = GELU()(current)
    elif activation == 'sigmoid':
        current = tf.keras.layers.Activation('sigmoid')(current)
    elif activation == 'tanh':
        current = tf.keras.layers.Activation('tanh')(current)
    elif activation == 'exp':
        current = Exp()(current)
    elif activation == 'softplus':
        current = Softplus()(current)
    else:
        print('Unrecognized activation "%s"' % activation, file=sys.stderr)
        exit(1)

    return current


###########################
# Keras blocks
###########################


def conv_block(inputs, filters=None, kernel_size=1, activation='relu', strides=1,
    dilation_rate=1, l2_scale=0, dropout=0, conv_type='standard', residual=False,
    pool_size=1, batch_norm=False, bn_momentum=0.99, bn_gamma=None,
    kernel_initializer='he_normal'):


    # flow through variable current
    current = inputs

    # choose convolution type
    if conv_type == 'separable':
        conv_layer = tf.keras.layers.SeparableConv1D
    else:
        conv_layer = tf.keras.layers.Conv1D

    if filters is None:
        filters = inputs.shape[-1]

    # activation
    current = activate(current, activation)

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = 'zeros' if residual else 'ones'

        current = tf.keras.layers.BatchNormalization(momentum=bn_momentum, gamma_initializer=bn_gamma, fused=True)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs,current])

    # Pool
    if pool_size > 1:
        current = tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')(current)

    return current


def conv_tower(inputs, filters_init, filters_mult=1, repeat=1, **kwargs):
    # flow through variable current
    current = inputs

    # initialize filters
    rep_filters = filters_init

    for ri in range(repeat):
        # convolution
        current = conv_block(current, filters=int(np.round(rep_filters)), **kwargs)

    # update filters
    rep_filters *= filters_mult

    return current


def dilated_residual(inputs, filters, kernel_size=3, rate_mult=2, conv_type='standard',
                     dropout=0, repeat=1, round=False, **kwargs):
    # flow through variable current
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for ri in range(repeat):
        # For skip connection purpose
        rep_input = current

        # dilate
        current = conv_block(current, filters=filters, kernel_size=kernel_size,
                             dilation_rate=int(np.round(dilation_rate)),
                             conv_type=conv_type, bn_gamma='ones', **kwargs)

        # return
        current = conv_block(current, filters=int(rep_input.shape[-1]), dropout=dropout, bn_gamma='zeros', **kwargs)

        # residual add
        current = tf.keras.layers.Add()([rep_input, current])

        # update dilation rate
        dilation_rate *= rate_mult

        if round:
            dilation_rate = np.round(dilation_rate)

    return current


# 2D related blocks

def concat_dist_2d(inputs, **kwargs):
    current = ConcatDist2D()(inputs)
    return current

def one_to_two(inputs, operation='mean', **kwargs):
    current = OneToTwo(operation)(inputs)
    return current


def conv_block_2d(inputs, filters=128, activation='relu', conv_type='standard',
    kernel_size=1, strides=1, dilation_rate=1, l2_scale=0, dropout=0, pool_size=1,
    batch_norm=False, bn_momentum=0.99, bn_gamma='ones', symmetric=False):

    """Construct a single 2D convolution block.   """

    # flow through variable current
    current = inputs

    # activation
    current = activate(current, activation)

  # choose convolution type
    if conv_type == 'separable':
        conv_layer = tf.keras.layers.SeparableConv2D
    else:
        conv_layer = tf.keras.layers.Conv2D

    # convolution
    current = conv_layer(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        dilation_rate=dilation_rate,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(l2_scale))(current)

    # batch norm
    if batch_norm:
        current = tf.keras.layers.BatchNormalization(
          momentum=bn_momentum,
          gamma_initializer=bn_gamma,
          fused=True)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # pool
    if pool_size > 1:
        current = tf.keras.layers.MaxPool2D(
          pool_size=pool_size,
          padding='same')(current)

    # symmetric
    if symmetric:
        current = layers.Symmetrize2D()(current)

    return current


def symmetrize_2d(inputs, **kwargs):
    return Symmetrize2D()(inputs)


def dilated_residual_2d(inputs, filters, kernel_size=3, rate_mult=2,
                        dropout=0, repeat=1, symmetric=True, **kwargs):
    """Construct a residual dilated convolution block.
    """

    # flow through variable current
    current = inputs

    # initialize dilation rate
    dilation_rate = 1.0

    for ri in range(repeat):
        rep_input = current

        # dilate
        current = conv_block_2d(current,
                                filters=filters,
                                kernel_size=kernel_size,
                                dilation_rate=int(np.round(dilation_rate)),
                                bn_gamma='ones',
                                **kwargs)

        # return
        current = conv_block_2d(current,
                                filters=int(rep_input.shape[-1]),
                                dropout=dropout,
                                bn_gamma='zeros',
                                **kwargs)

        # residual add
        current = tf.keras.layers.Add()([rep_input, current])

        # enforce symmetry
        if symmetric:
            current = Symmetrize2D()(current)

        # update dilation rate
        dilation_rate *= rate_mult

    return current


def cropping_2d(inputs, cropping, **kwargs):
    current = tf.keras.layers.Cropping2D(cropping)(inputs)
    return current


def upper_tri(inputs, diagonal_offset=2, **kwargs):
    current = UpperTri(diagonal_offset)(inputs)
    return current


def dense(inputs, units, activation='softplus', kernel_initializer='he_normal',
          l2_scale=0, l1_scale=0, **kwargs):

    current = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale)
    )(inputs)

    return current



# Main class
class Model:
    def __init__(self):
        SEQ_LENGTH = 524288
        sequence = tf.keras.Input(shape=(SEQ_LENGTH, 4), name='sequence')

        current = sequence

        # Augmentation - Enable it later (after good performance on the training set)
        # current, reverse_bool = StochasticReverseComplement()(current)
        # augment_shift = 11
        # current = StochasticShift(augment_shift)(current)

        # TRUNK:

        # First 1D convolution
        current = conv_block(current, filters=96, kernel_size=11, pool_size=2, batch_norm=True, bn_momentum=0.9265,
                             activation="relu")


        # Change for repeat = 9

        # Multiple (11) 1D convolutions in a tower to arrive to 2048bp representation in 1D vectors
        current = conv_tower(current, filters_init=96, filters_mult=1.0, kernel_size=5, pool_size=2, repeat=9,
                             batch_norm=True, bn_momentum=0.9265, activation="relu")

        # Dilated residual layers
        current = dilated_residual(current, filters=48, rate_mult=1.75, repeat=8, dropout=0.4, batch_norm=True,
                                   bn_momentum=0.9265, activation="relu")

        # Bottleneck 1D convolution
        current = conv_block(current, filters=64, kernel_size=5, batch_norm=True, bn_momentum=0.9265,
                             activation="relu")

        # final activation
        current = activate(current, "relu")


        # HEAD:
        current = one_to_two(current)
        current = concat_dist_2d(current)
        current = conv_block_2d(current, filters=48, kernel_size=3, batch_norm=True, bn_momentum=0.9265,
                             activation="relu")

        current = symmetrize_2d(current)
        current = dilated_residual_2d(current, filters=24, kernel_size=3, rate_mult=1.75, repeat=6, dropout=0.1,
                                      batch_norm=True, bn_momentum=0.9265, activation="relu")

        # TODO: TRY WITHOUT CROP
        #current = cropping_2d(current, cropping=26)

        current = upper_tri(current, diagonal_offset=2)

        current = dense(current, units=1, activation="linear")
        #current = dense(current, units=5, activation="linear")

        # current = SwitchReverse()([current, reverse_bool])

        # make model
        self.model = tf.keras.Model(inputs=sequence, outputs=current)


    def get_model(self):
        return self.model