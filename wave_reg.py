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

def alt_second_order_c_known(f, xyz, tape):
    #x, y, t = xyz
    df_dxyt = tape.batch_jacobian(f, xyz)
    df_dxyt2 = tape.batch_jacobian(df_dxyt, xyz)

    fxx = df_dxyt2[..., 0, 0]
    fyy = df_dxyt2[..., 1, 1]
    ftt = df_dxyt2[..., 2, 2]
    del_f = fxx + fyy

    # TODO: c might not be 1
    c_sqd = 1
    # TODO: The source ISN'T zero
    source_t = 0
    wave_eq = ftt - c_sqd * del_f - source_t

    grad_loss = tf.math.reduce_mean(tf.math.abs(wave_eq))

    return grad_loss

def first_order(f, xyz, tape):
    x, y, t = xyz
    p, u, v = tf.unstack(f, axis=1)

    p_t = nth_gradient(p, t, 1, tape)
    div_vel = nth_gradient(u, x, 1, tape) + nth_gradient(v, y, 1, tape)

    p_x = nth_gradient(p, x, 1, tape)
    p_y = nth_gradient(p, y, 1, tape)
    u_t = nth_gradient(u, t, 1, tape)
    v_t = nth_gradient(v, t, 1, tape)

    # TODO: Constants are medium dependent
    # TODO: Source can change
    kappa = 1
    rho = 1
    source = 0

    first_eq = source - p_t - kappa*div_vel
    second_eq_l1_norm = tf.math.abs(u_t + (1/rho)*p_x) + tf.math.abs(v_t + (1/rho)*p_y) 

    eq_sys = tf.concat([first_eq, second_eq_l1_norm], axis=0)
    grad_loss = tf.math.reduce_mean(tf.math.abs(eq_sys))
    return grad_loss

def first_order_curl(f, xyz, tape):
    x, y, t = xyz
    p, u, v = tf.unstack(f, axis=1)
    
    p_t = nth_gradient(p, t, 1, tape)
    div_vel = nth_gradient(u, x, 1, tape) + nth_gradient(v, y, 1, tape)

    u_t = nth_gradient(u, t, 1, tape)
    v_t = nth_gradient(v, t, 1, tape)
    u_ty = nth_gradient(u_t, y, 1, tape)
    v_tx = nth_gradient(v_t, x, 1, tape)

    # TODO: Constants are medium dependent
    # TODO: Source can change
    kappa = 1
    #rho = 1
    source = 0

    first_eq = source - p_t - kappa*div_vel
    second_eq_curl = v_tx - u_ty

    eq_sys = tf.concat([first_eq, second_eq_curl], axis=0)

    grad_loss = tf.math.reduce_mean(tf.math.abs(eq_sys))
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
    if regularizer == "first_explicit":
        return first_order
    elif regularizer == "first_curl":
        return first_order_curl
    elif regularizer in ["second_explicit", "second_c_known"]:
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
    
        