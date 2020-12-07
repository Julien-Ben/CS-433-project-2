import tensorflow.keras.backend as K

ALPHA = 0.6  # Closer to 1 will penalize False Negatives more (= Saying background when it's road)
BETA = 1 - ALPHA  # Closer to 1 will penalize False Positives more (= Saying road when it's background)
GAMMA = 0.75  # Non-linearity. Above one will focus on harder examples
# ALPHA = 0.5 and GAMMA = 1 is dice loss


def focal_tversky_loss(y_true, y_pred, gamma=GAMMA, alpha=ALPHA):
    """
    Non-linear Tversky loss
    """
    return K.pow((1 - tversky_index(y_true, y_pred, alpha=alpha)), gamma)


def tversky_loss(y_true, y_pred, alpha=ALPHA):
    """
    Weighted Dice loss
    """
    return 1 - tversky_index(y_true, y_pred, alpha=alpha)


def dice_loss(y_true, y_pred):
    """
    F1-score but as a loss function
    """
    return 1 - dice_coefficient(y_true, y_pred)


def dice_coefficient(y_true, y_pred, smooth=1):
    """
    F1-score
    """
    return tversky_index(y_true, y_pred, smooth=smooth, alpha=0.5)


def tversky_index(y_true, y_pred, smooth=1, alpha=ALPHA):
    """
    Computes the Tversky index, which is a weighted dice index
    Higher alpha means more penalty for false negatives
    """
    beta = 1 - alpha
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    fp = K.sum((1 - y_true_f) * y_pred_f)
    return (tp + smooth) / (tp + alpha*fn + beta*fp + smooth)

# U-net 2 currently trained with dice_coef as metric
# and dice_coef_loss as loss
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
    