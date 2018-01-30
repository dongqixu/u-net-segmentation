import tensorflow as tf

''' Loss Function Definition'''


# TODO: design new loss function
# self-define dice loss, useless interface removed
def dice_loss_function(prediction, ground_truth, *interface):
    ground_truth = tf.one_hot(indices=ground_truth, depth=3)
    dice_loss = 0
    for i in range(1, 3):
        # reduce_mean calculation -> reduce_sum
        intersection = tf.reduce_sum(prediction[:, :, :, :, i] * ground_truth[:, :, :, :, i])
        union_prediction = tf.reduce_sum(prediction[:, :, :, :, i] * prediction[:, :, :, :, i])
        union_ground_truth = tf.reduce_sum(ground_truth[:, :, :, :, i] * ground_truth[:, :, :, :, i])
        union = union_ground_truth + union_prediction
        # weight should sum to 1 -> sum as 2 -> OK
        weight = 1 - (tf.reduce_sum(ground_truth[:, :, :, :, i]) / tf.reduce_sum(ground_truth))
        dice_loss += (1 - 2 * intersection / union) * weight
    return dice_loss


# TODO: all zero cases -> rearrange weight? is it important?
# TODO: may need to update weight
def softmax_loss_function(prediction, ground_truth):
    # loss = weight * - target * log(softmax(logits))
    # prediction = logits
    softmax_prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=ground_truth, depth=3)
    loss = 0
    for i in range(1, 3):
        class_i_ground_truth = ground_truth[:, :, :, :, i]
        class_i_prediction = softmax_prediction[:, :, :, :, i]
        # weight should sum to 1 -> sum as 2 -> OK
        # weight = 1 - (tf.reduce_sum(class_i_ground_truth) / tf.reduce_sum(ground_truth))
        weight = 1.0  # equal weight
        loss = loss - tf.reduce_mean(weight * class_i_ground_truth * tf.log(
            tf.clip_by_value(t=class_i_prediction, clip_value_min=0.005, clip_value_max=1)))
        # Clips tensor values to a specified min and max.
    return loss


'''
# interface no long use
def get_weight_list(ratio):
    ratio_x, ratio_y, ratio_z = ratio
    weight_x = 1 - ratio_x
    weight_y = ratio_x / (ratio_y + 1e-5)
    weight_z = ratio_x / (ratio_z + 1e-5)
    weight_y = tf_log10(weight_y) + (1 - ratio_y) * 0.5
    weight_z = tf_log10(weight_z) + (1 - ratio_z) * 0.5
    weight_sum = weight_x + weight_y + weight_z
    weight_x, weight_y, weight_z = \
        2 * weight_x / weight_sum, 2 * weight_y / weight_sum, 2 * weight_z / weight_sum
    return weight_x, weight_y, weight_z


def tf_log10(tensor):
    numerator = tf.log(tensor)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator
'''
