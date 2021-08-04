from targets import nth_gradient
import tensorflow as tf
from helpers import GradReg

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

    return wave_eq

def velocity_lr(f, xyz, tape):
    p, u, v = tf.unstack(f, axis=1)
    return u

def velocity_ud(f, xyz, tape):
    p, u, v = tf.unstack(f, axis=1)
    return v

def second_curl(f, xyz, tape):
    x, y, t = xyz
    p, u, v = tf.unstack(f, axis=1)

    u_t = nth_gradient(u, t, 1, tape)
    v_t = nth_gradient(v, t, 1, tape)
    u_ty = nth_gradient(u_t, y, 1, tape)
    v_tx = nth_gradient(v_t, x, 1, tape)

    curly = v_tx - u_ty
    return curly

def first_order_c_known(f, xyz, tape):
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
    second_eq_A = u_t + (1/rho)*p_x
    second_eq_B = v_t + (1/rho)*p_y
    eqs = tf.stack([first_eq, second_eq_A, second_eq_B], axis=-1)
    return eqs

def second_order_c_known(f, xyz, tape):
    # raise ValueError("needs to be updated")
    # Unpack the columns
    x,y,t = xyz
    puv = tf.unstack(f, axis=1)
    p = puv[0]

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    ftt = nth_gradient(f, t, 2, tape)
    del_f = fxx + fyy
    # TODO: c might not be 1
    c_sqd = 1
    # TODO: The source ISN'T zero
    source_t = 0
    wave_eq = ftt - c_sqd * del_f - source_t

    return wave_eq


def second_order_c_known_late_start(f, xyz, tape):
    raise ValueError("needs to be updated")
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
    raise ValueError("needs to be updated")
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
    raise ValueError("needs to be updated")
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
    raise ValueError("needs to be updated")
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

name_to_reg_by_category = {
    "wave_eq": {
        "first_explicit": {"interior": first_order_c_known},
        "second_explicit": {"interior": second_order_c_known},
        "second_c_known": {"interior": second_order_c_known},
        "second_explicit_late": {"interior": second_order_c_known_late_start},
        "second_explicit_no_middle": {"interior": second_order_c_known_no_middle},
        "second_implicit": {"interior": second_order_c_unknown},
        "second_c_unknown": {"interior": second_order_c_unknown},
        "third_implicit": {"interior": third_order_c_unknown},
        "third_c_unknown": {"interior": third_order_c_unknown},
        "second_curl": {"interior": second_curl},
    },
    "boundary": {
        "velocity": {"boundary_lr": velocity_lr, "boundary_ud": velocity_ud}
    }
}

name_to_reg = {
    "first_explicit": first_order_c_known,
    "second_explicit": second_order_c_known,
    "second_c_known": second_order_c_known,
    "second_explicit_late": second_order_c_known_late_start,
    "second_explicit_no_middle": second_order_c_known_no_middle,
    "second_implicit": second_order_c_unknown,
    "second_c_unknown": second_order_c_unknown,
    "third_implicit": third_order_c_unknown,
    "third_c_unknown": third_order_c_unknown,
    "second_curl": second_curl,
    "velocity_lr": velocity_lr,
    "velocity_ud": velocity_ud,
}

def clean_append(my_dict, key, element):
    if key in my_dict:
        my_dict.append(element)
    else:
        my_dict[key] = [element]

def get_wave_reg_old(gr_names_by_category):

    # Ensure grad_reg_names is a list
    if not isinstance(gr_names_by_category, dict):
        raise ValueError("GR Names must be a dictionary")

    grad_regs = {}

    for category, gr_names in gr_names_by_category.items():
        name_to_reg = name_to_reg_by_category[category]
        for name in gr_names:
            reg = name_to_reg[name]
            for region, func in reg.items():
                clean_append(grad_regs, region, func)

    return grad_regs



def get_wave_grs(gr_dicts):
    if gr_dicts == "none":
        return []
    else:
        grs = []
        for gr_dict in gr_dicts:
            # Add the vector function
            name = gr_dict['name']
            gr_dict['vector_func'] = name_to_reg[name]
            gr = GradReg(gr_dict)
            grs.append(gr)
        return grs

        