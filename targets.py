import tensorflow as tf

def nth_gradient(y, x, n, tape):
	'''
	Compute the nth order gradient of y wrt x (using tape)
	'''
	grad = y
	for i in range(n):
		grad = tape.gradient(grad, x)
	return grad

############
# Parabola #
############

def parabola(x, y, f_a=1.0, f_b=1.0):
	'''
	Your friendly neighborhood parabola.
	'''
	# Function coefficients (f = f_a*x^2 + f_b*y^2)
	return f_a * x**2 + f_b * y**2	# f_a*x^2 + f_b*y^2
        
def parabola_regularizer_zero(f, xyz, tape):
    '''
    Parabola gradient condition.
    Third order derivs = 0
    '''

    # Unpack the columns
    x,y = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    fxxy = tape.gradient(fxx, y)
    fyyx = tape.gradient(fyy, x)
    fxxx = tape.gradient(fxx, x)
    fyyy = tape.gradient(fyy, y)

    # L1 regularizer
    grads = tf.concat([fxxx, fxxy, fyyx, fyyy], axis=0)
    grad_loss = tf.math.reduce_mean(tf.math.abs(grads))

    return grad_loss

def parabola_regularizer_const(f, xyz, tape):
    '''
    Variance of second order derivs = 0
    '''

    # Unpack the columns
    x,y = xyz

    fx = nth_gradient(f,x,1,tape)
    fy = nth_gradient(f,y,1,tape)
    fxx = tape.gradient(fx, x)
    fxy = tape.gradient(fx, y)
    fyy = tape.gradient(fy, y)

    grad_loss = (tf.math.reduce_variance(fxx) + tf.math.reduce_variance(fxy)
                + tf.math.reduce_variance(fyy))
            
    return grad_loss

def parabola_regularizer_first(f, xyz, tape):
    '''
    Variance of first order derivs divided by vals = 0
    '''

    x,y = xyz

    fx = nth_gradient(f,x,1,tape)
    fy = nth_gradient(f,y,1,tape)

    grad_loss = (tf.math.reduce_variance(tf.math.divide_no_nan(fx,x)) 
                + tf.math.reduce_variance(tf.math.divide_no_nan(fy,y)))

    return 0.5*grad_loss
    

##############
# Oscillator #
##############

def waveish(x, y, k=3):
	'''
    Ye olde solution to the Laplace equation.
	'''
	# Function coefficients (f = f_a*x^2 + f_b*y^2)
	return tf.math.sin(k*x) * tf.math.cosh(k*y)

def waveish_regularizer_zero(f, xyz, tape):
    '''
    GL = fxx + fyy
    '''

    x, y = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)

    # L1 regularizer
    grads = tf.concat([fxx, fyy], axis=0)
    grad_loss = tf.math.reduce_mean(tf.math.abs(grads))

    return grad_loss

##########
# Cosine #
##########

#########
# Cubic #
#########

def cubic(x, y, f_a=1.0, f_b=1.0):
	'''
	Your not so friendly neighborhood cubic.
	'''
	# Function coefficients (f = f_a*x^2 + f_b*y^2)
	return f_a * x**3 + f_b * y**3	# f_a*x^2 + f_b*y^2
        
def cubic_regularizer_zero(f, xyz, tape):
    '''
    Parabola gradient condition.
    Third order derivs = 0
    '''

    # Unpack the columns
    x,y = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    fxxx = tape.gradient(fxx, x)
    fxyy = tape.gradient(fyy, x)
    fyyy = tape.gradient(fyy,y)
    fxxxx = tape.gradient(fxxx,x)
    fxxxy = tape.gradient(fxxx,y)
    fxxyy = tape.gradient(fxyy,x)
    fxyyy = tape.gradient(fxyy,y)
    fyyyy = tape.gradient(fyyy,y)

    # L1 regularizer
    grads = tf.concat([fxxxx, fxxyy, fxxyy, fxyyy, fyyyy], axis=0)
    grad_loss = tf.math.reduce_mean(tf.math.abs(grads))

    return grad_loss

def cubic_regularizer_const(f, xyz, tape):
    '''
    Variance of second order derivs = 0
    '''

    # Unpack the columns
    x,y = xyz

    fxx = nth_gradient(f, x, 2, tape)
    fyy = nth_gradient(f, y, 2, tape)
    fxxy = tape.gradient(fxx, y)
    fxyy = tape.gradient(fyy, x)
    fxxx = tape.gradient(fxx, x)
    fyyy = tape.gradient(fyy, y)

    grad_loss = (tf.math.reduce_variance(fxxx) + 3*tf.math.reduce_variance(fxxy)
                + 3*tf.math.reduce_variance(fxyy) + tf.math.reduce_variance(fyyy))
            
    return grad_loss

##################################
##################################

def get_target(name: str, regularizer: str, configs):
    if name == 'parabola':
        f_a, f_b = configs.target_coefficients
        f = lambda x,y : parabola(x,y, f_a, f_b)
        if regularizer in ["zero", "third"]:
            reg = parabola_regularizer_zero
        elif regularizer in ["const", "second"]:
            reg = parabola_regularizer_const
        elif regularizer == "none":
            reg = None
        elif regularizer == "first":
            reg = parabola_regularizer_first
        else:
            raise ValueError("Unknown regularizer for " + name + ", " + regularizer)

    elif name == 'cubic':
        f_a, f_b = configs.target_coefficients
        f = lambda x,y : cubic(x,y, f_a, f_b)
        if regularizer in ["zero", "fourth"]:
            reg = cubic_regularizer_zero
        elif regularizer in ["const", "third"]:
            reg = cubic_regularizer_const
        elif regularizer == "none":
            reg = None
        else:
            raise ValueError("Unknown regularizer for " + name + ", " + regularizer)
    
    elif name == "sin-cosh":
        # Unpack coefficients
        k, = configs.target_coefficients[0]
        f = lambda x,y : waveish(x, y, k=k)
        if regularizer in ["zero", "laplacian"]:
            reg = waveish_regularizer_zero
        else:
            raise ValueError("Unknown regularizer for " + name + ", " + regularizer)
    
    else:
        raise ValueError("Unknown target function " + name)

    return f, reg


# def partial_derivatives(f, xyz, n, tape):
#     '''
#     Computes every unique nth partial derivative of f wrt. xyz
#     Parameters:
#     f - Nx1 Tensor of outputs of f
#     xyz - list of Nx1 Tensors, input columns
#     Example: If xyz = [x,y,z] and n = 2, returns [fxx | fxy | fyy | fyz | fzz | fxz]
#     '''
#     partial_list = [f]
#     for i in range(n):
#         grads = [nth_gradient(f, x) for x in xyz]
#     partials = tf.stack(partial_list, axis=1)
#     return partials


# def all_partials(f, xyz, n, tape):
#     if n == 0:
#         raise ValueError("Why are you doing this...")
#     elif n == 1:
#         return [tape.gradient(f, x) for x in xyz]
#     elif n == 2:
#         fx = nth_gradient(f,x,1,tape)
#         fy = nth_gradient(f,y,1,tape)
#         fxx = tape.gradient(fx, x)
#         fxy = tape.gradient(fx, y)
#         fyy = tape.gradient(fy, y)
#         return [fxx, fxy, fyy]
#     elif n == 3:


# def parabola_condition_zero(f, xyz, tape):
#     f_partials = all_partials(f, xyz, 3, tape)
#     f_partials = tf.concat(f_partials, axis=0)
#     grad_loss = tf.math.reduce_mean(tf.math.abs(f_partials))
#     return grad_loss