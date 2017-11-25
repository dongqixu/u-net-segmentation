import tensorflow as tf

''' Loss Function Definition'''


# self-define dice loss
def dice_loss_function(prediction, ground_truth):
    # TODO: any influence on softmax?
    # prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=ground_truth, depth=3)
    dice_loss = 0
    for i in range(3):
        # reduce_mean calculation -> reduce_sum
        intersection = tf.reduce_sum(prediction[:, :, :, :, i] * ground_truth[:, :, :, :, i])
        union_prediction = tf.reduce_sum(prediction[:, :, :, :, i] * prediction[:, :, :, :, i])
        union_ground_truth = tf.reduce_sum(ground_truth[:, :, :, :, i] * ground_truth[:, :, :, :, i])
        union = union_ground_truth + union_prediction
        # weight should sum to 1 -> sum as 2 -> OK
        weight = 1 - (tf.reduce_sum(ground_truth[:, :, :, :, i]) / tf.reduce_sum(ground_truth))
        dice_loss += (1 - 2 * intersection / union) * weight
    return dice_loss


# TODO: all zero cases -> rearrange wegiht? is it important?
def softmax_loss_function(prediction, ground_truth):
    # loss = weight * - target * log(softmax(logits))
    # prediction = logits
    softmax_prediction = tf.nn.softmax(logits=prediction)
    ground_truth = tf.one_hot(indices=ground_truth, depth=3)
    loss = 0
    for i in range(3):
        class_i_ground_truth = ground_truth[:, :, :, :, i]
        class_i_prediction = softmax_prediction[:, :, :, :, i]
        # weight should sum to 1 -> sum as 2 -> OK
        weight = 1 - (tf.reduce_sum(class_i_ground_truth) / tf.reduce_sum(ground_truth))
        loss = loss - tf.reduce_mean(weight * class_i_ground_truth * tf.log(
            tf.clip_by_value(t=class_i_prediction, clip_value_min=0.005, clip_value_max=1)))
        # Clips tensor values to a specified min and max.
    return loss
