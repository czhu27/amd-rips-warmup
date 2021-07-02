import tensorflow as tf

def parabola(x, y, f_a=1.0, f_b=1.0):
	'''
	Your friendly neighborhood parabola.
	'''
	# Function coefficients (f = f_a*x^2 + f_b*y^2)
	return f_a * x**2 + f_b * y**2	# f_a*x^2 + f_b*y^2

def parabola_condition_const():
    '''
    Vector of gradient calculations that should be constant
    '''
    return nth_gradient(f, xy, 2, tape)

def nth_gradient(y, x, n, tape):
	'''
	Compute the nth order gradient of y wrt x (using tape)
	'''
	grad = y
	for i in range(n):
		grad = tape.gradient(grad, x)
	return grad

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




def gradient_condition(f, x, y, tape):
	'''
	Parabola gradient condition
	'''
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

def get_target(name: str, regularizer: str, configs):
    if name == 'parabola':
        f = lambda x,y : parabola(x,y, configs.f_a, configs.f_b)
        if regularizer == "const":
            cond = parabola_condition_const()
            
    else:
        raise ValueError("Unknown target function " + name)
    
    def grad_reg():
        cond = parabola_condition_const()
        return zero_variance(cond)

    return f, grad_reg