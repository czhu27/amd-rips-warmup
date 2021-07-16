from targets import nth_gradient
import tensorflow as tf

def second_order_c_known(f, xyz, tape):
    # Unpack the columns
    x,y,t = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    ftt = nth_gradient(f, t, 2, tape)
    del_f = fxx + fyy
    # TODO: c might not be 1
    c_sqd = 1
    # TODO: The source ISN'T zero
    source_t = 0
    wave_eq = ftt - c_sqd * del_f - source_t

    # L1 regularizer
    #grads = tf.concat([fxxx, fxxy, fyyx, fyyy], axis=0)
    grad_loss = tf.math.reduce_mean(tf.math.abs(wave_eq))

    return grad_loss

def second_order_c_known_L2(f, xyz, tape):
    # Unpack the columns
    x,y,t = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    ftt = nth_gradient(f, t, 2, tape)
    del_f = fxx + fyy
    # TODO: c might not be 1
    c_sqd = 1
    # TODO: The source ISN'T zero
    source_t = 0
    wave_eq = ftt - c_sqd * del_f - source_t

    # L1 regularizer
    #grads = tf.concat([fxxx, fxxy, fyyx, fyyy], axis=0)
    grad_loss = tf.math.reduce_mean(wave_eq ** 2)

    return grad_loss


def second_order_c_known_late_start(f, xyz, tape):
    # Unpack the columns
    x,y,t = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    ftt = nth_gradient(f, t, 2, tape)
    del_f = fxx + fyy
    # TODO: c might not be 1
    c_sqd = 1
    # TODO: The source ISN'T zero
    source_t = 0
    wave_eq = ftt - c_sqd * del_f - source_t

    start_time = 0.2
    wave_eq = wave_eq[t >= start_time]

    # L1 regularizer
    #grads = tf.concat([fxxx, fxxy, fyyx, fyyy], axis=0)
    grad_loss = tf.math.reduce_mean(tf.math.abs(wave_eq))

    return grad_loss


def second_order_c_known_no_middle(f, xyz, tape):
    # Unpack the columns
    x,y,t = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    ftt = nth_gradient(f, t, 2, tape)
    del_f = fxx + fyy
    # TODO: c might not be 1
    c_sqd = 1
    # TODO: The source ISN'T zero
    source_t = 0
    wave_eq = ftt - c_sqd * del_f - source_t

    start_time = 0.3
    x0, y0 = 0.5, 0.5
    radius = 0.1
    cond = (x <= x0 + radius) & (x >= x0 - radius) & (y <= y0 + radius) & (y >= y0 - radius)
    wave_eq = wave_eq[cond]

    # L1 regularizer
    #grads = tf.concat([fxxx, fxxy, fyyx, fyyy], axis=0)
    grad_loss = tf.math.reduce_mean(tf.math.abs(wave_eq))

    return grad_loss

def second_order_c_unknown(f, xyz, tape):
    # Unpack the columns
    x,y,t = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    ftt = nth_gradient(f, t, 2, tape)
    del_f = fxx + fyy
    # TODO: The source ISN'T zero
    source_t = 0
    c_sqd = tf.math.divide_no_nan(ftt - source_t, del_f)

    grad_loss = tf.math.reduce_variance(c_sqd)

    return grad_loss

def third_order_c_unknown(f, xyz, tape):
    # Unpack the columns
    x,y,t = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    ftt = nth_gradient(f, t, 2, tape)
    fxxt = tape.gradient(fxx, t)
    fyyt = tape.gradient(fyy, t)
    fttt = tape.gradient(ftt, t)
    del_f = fxx + fyy
    # TODO: The source ISN'T zero
    source_t = 0
    c_sqd = tf.math.divide_no_nan(ftt - source_t, del_f)

    grad_loss = tf.math.reduce_variance(c_sqd)

    return grad_loss


def get_wave_reg(regularizer, configs):
    if regularizer in ["second_explicit", "second_c_known"]:
        return second_order_c_known
    elif regularizer == "second_explicit_L2":
        return second_order_c_known_L2
    elif regularizer == "second_explicit_late":
        return second_order_c_known_late_start
    elif regularizer == "second_explicit_no_middle":
        return second_order_c_known_no_middle
    elif regularizer in ["second_implicit", "second_c_unknown"]:
        return second_order_c_unknown
    elif regularizer in ["third_implicit", "third_c_unknown"]:
        return third_order_c_unknown
    elif regularizer == "none":
        return None
    else:
        raise ValueError(f"Unknown regularizer {regularizer} for wave eq.")
    
        