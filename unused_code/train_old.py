from wave_data import data_wave
def main(configs: Configs):
	# Setup folder structure vars
	output_dir = configs.output_dir
	# TB logs
	log_dir = output_dir + "/logs" 
	# TODO: This is a hack. Understand tensorboard dirs better...
	scalar_dir = log_dir + "/train" #+ "/scalars"
	metrics_dir = scalar_dir #+ "/metrics"
	figs_folder = output_dir + "/figs"
	results_dir = output_dir + "/results"
	os.makedirs(figs_folder, exist_ok=True)
	os.makedirs(output_dir, exist_ok=True)
	print("Saving to output dir: ", output_dir)

	# Save configs
	yaml.safe_dump(configs.__dict__, open(output_dir + "/configs.yaml", "w"))

	# Setup Tensorboard
	file_writer = tf.summary.create_file_writer(metrics_dir)
	file_writer.set_as_default()

	# ------------------------------------------------------------------------------
	# General setup
	# ------------------------------------------------------------------------------
	# Set seeds for reproducibility
	np.random.seed(configs.seed)
	tf.random.set_seed(configs.seed)

	# ------------------------------------------------------------------------------
	# Data preparation
	# ------------------------------------------------------------------------------
	# Data for training NN based on L_f loss function
	#X_f_l, X_f_ul = data_creation(configs.dataset, configs.corners)
	
	X_w_l, X_w_ul, Y_l, x_flat, y_flat, t_flat, p_flat  = data_wave(50, 10, 10, 1, [8000, 4000, 5000, 1.0, 1.0, 0.0])
	# Set target function
	#f, grad_reg = get_target(configs.target, configs.gradient_loss, configs)
	# f = lambda x,y : parabola(x,y, configs.f_a, configs.f_b)

	#f_true = f(X_f_l[:, 0:1], X_f_l[:, 1:2])
	#f_ul = tf.zeros((X_f_ul.shape[0], 1))
	f_true = p_flat

	if configs.noise > 0:
		f_true += np.reshape(configs.noise*np.random.randn((len(f_true))),(len(f_true),1))
		
	# is_labeled_l = tf.fill(f_true.shape, True)
	# is_labeled_ul = tf.fill(f_ul.shape, False)
	is_labeled_l = tf.fill(X_w_l.shape, True)
	is_labeled_ul = tf.fill(X_w_ul.shape, False)
	

	X_f_all = tf.concat([x_flat, y_flat, t_flat], 1)
	f_all = p_flat #tf.concat([f_true, f_ul], axis=0)
	is_labeled_all = tf.concat([is_labeled_l, is_labeled_ul], axis=0)
	#is_labeled_all = tf.concat([is_labeled_l, is_labeled_ul], axis=0)

	# if "data-distribution" in configs.plots:
	# 	print("Saving data distribution plots")
	# 	plot_data(X_f_l, "labeled", figs_folder)
	# 	plot_data(X_f_ul, "unlabeled", figs_folder)
	# 	plot_data(X_f_all, "all", figs_folder)
	

	# Create TensorFlow dataset for passing to 'fit' function (below)
	dataset = tf.data.Dataset.from_tensors((X_f_all, f_all, is_labeled_all))

	# ------------------------------------------------------------------------------
	# Create neural network (physics-inspired)
	# ------------------------------------------------------------------------------
	layers = configs.layers
	model = create_nn(layers, configs)
	model.summary()

	# TODO: Hacky add...
	#model.gradient_regularizer = grad_reg

	# ------------------------------------------------------------------------------
	# Assess accuracy with non-optimized model
	# ------------------------------------------------------------------------------
	f_pred_0 = model.predict(X_f_all)
	#error_0 = np.sqrt(np.mean(np.square(f_pred_0 - np.reshape(p, (500,2500,9)))))

	# ------------------------------------------------------------------------------
	# Model compilation / training (optimization)
	# ------------------------------------------------------------------------------
	if configs.lr_scheduler:
		opt_step = tf.keras.optimizers.schedules.PolynomialDecay(
			configs.lr_scheduler_params[0], configs.lr_scheduler_params[2], 
			end_learning_rate=configs.lr_scheduler_params[1], power=configs.lr_scheduler_params[3],
			cycle=False, name=None) #Changing learning rate
	else:
		print(type(configs.lr))
		if not isinstance(configs.lr, float):
			raise ValueError("configs.lr must be floats (missing a decimal point?)")
		opt_step = configs.lr		# gradient descent step

	opt_batch_size = configs.batch_size	# batch size
	opt_num_its = configs.epochs		# number of iterations

	model.set_batch_size(opt_batch_size)

	optimizer = optimizers.Adam(learning_rate = opt_step)
	model.compile(optimizer = optimizer, run_eagerly=configs.debug)		# DEBUG
	tic = time.time()

	# Define Tensorboard Callbacks
	class TimeLogger(keras.callbacks.Callback):
		def __init__(self):
			pass
		def on_train_begin(self, logs):
			self.train_start = time.time()
		def on_epoch_begin(self, epoch, logs):
			self.epoch_start = time.time()
		def on_epoch_end(self, epoch, logs=None):
			train_dur = time.time() - self.train_start
			epoch_dur = time.time() - self.epoch_start
			tf.summary.scalar('Time/Total', data=train_dur, step=epoch)
			tf.summary.scalar('Time/Epoch', data=epoch_dur, step=epoch)

	class StressTestLogger(keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs):
			self.test_every = 100
			if epoch % self.test_every == self.test_every - 10:
				# Make grid to display true function and predicted
				error1 = compute_error(model, f, -1.0, 1.0)
				tf.summary.scalar('Error/interpolation', data=error1, step=epoch)
				error2 = compute_error(model, f, -2.0, 2.0)
				tf.summary.scalar('Error/extrapolation', data=error2, step=epoch)

	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	logging_callbacks = [TimeLogger(), StressTestLogger(), tensorboard_callback]

	if "tensorboard" in configs.plots:
		print("Using tensorboard callbacks")
		callbacks = logging_callbacks
	else:
		callbacks = []

	model.fit(dataset, 
			epochs=opt_num_its, 
			verbose=2,
			callbacks=callbacks)
	toc = time.time()
	print("Training time: {:.2F} s\n".format(toc - tic))

	if "model" in configs.saves:
		print("Saving final model")
		model.save(output_dir + "/model")

	# ------------------------------------------------------------------------------
	# Assess accuracy with optimized model and compare with non-optimized model
	# ------------------------------------------------------------------------------
	#f_pred_1 = model.predict(X_f_l)
	f_pred_1 = model.predict(X_w_l)
	#error_1 = np.sqrt(np.mean(np.square(f_pred_1 - f_true)))
	error_1 = np.sqrt(np.mean(np.square(f_pred_1 - Y_l)))
	loss_value = model.loss_function_f(f_pred_1, Y_l)/X_w_l.shape[0]

	#print("Train set error (before opt): {:.15E}".format(error_0))
	print("Train set error (after opt) : {:.15E}".format(error_1))
	#print("Ratio of errors             : {:.1F}".format(error_0/error_1))
	print("Loss function value         : {:.15E}".format(loss_value))

	# ------------------------------------------------------------------------------
	# Stress set - Assess extrapolation capabilities
	# ------------------------------------------------------------------------------

	# Make grid to display true function and predicted
	#error1 = compute_error(model, f, -1.0, 1.0)
	#print("Error [-1,1]x[-1,1] OLD: {:.6E}".format(error1))
	##error2 = compute_error(model, f, -2.0, 2.0)
	#error2 = extrap_error(model, f, -1.0, 1.0, -2.0, 2.0)
	#print("Error [-2,2]x[-2,2]: {:.6E}".format(error2))
	#error3 = compute_error(model, f, -3.0, 3.0)
	#error3 = extrap_error(model, f, -2.0, 2.0, -3.0, 3.0)
	#print("Error [-3,3]x[-3,3]: {:.6E}".format(error3))

	# if "extrapolation" in configs.plots:
	# 	print("Saving extrapolation plots")
	# 	buf = plot_gridded_functions(model, f, -1.0, 1.0, "100", folder=figs_folder)
	# 	buf = plot_gridded_functions(model, f, -2.0, 2.0, "200", folder=figs_folder)
	# 	buf = plot_gridded_functions(model, f, -3.0, 3.0, "300", folder=figs_folder)

	if "extrapolation_wave" in configs.plots:
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		nx = ny = 10
		dt = .02
		t = 0

		def update(frame, fig, dt, nx, ny):
			X = np.arange(0,1,1/nx)
			Y = np.arange(0,1,1/ny)
			X, Y = np.meshgrid(X,Y)
			grid_pts = np.reshape(np.concatenate((X,Y)), (nx*ny,2))
			time_vec = np.ones((len(grid_pts),1))*dt*frame
			print(time_vec)
			inputs = np.concatenate((grid_pts,time_vec), axis=1)
			soln = model.predict(inputs)
			soln = np.reshape(soln, (nx,ny))
			print(soln)


			if len(fig.axes[0].collections) != 0:
				fig.axes[0].collections = []
				surf = fig.axes[0].plot_surface(X, Y, soln, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			else:
				surf = fig.axes[0].plot_surface(X, Y, soln, cmap=cm.coolwarm, linewidth=0, antialiased=False)
			ax.set_zlim(-.04, .04)

			fig.canvas.draw()
			return surf,

		ani = FuncAnimation(fig, update, fargs=[fig, dt, nx, ny], frames=50, blit=True)
		ani.save('wave.gif', writer = 'PillowWriter', fps=5)

		# print("Saving extrapolation plots")
		# buf = make_movie(model, f, -1.0, 1.0, "100", folder=figs_folder)
		# buf = make_movie(model, f, -2.0, 2.0, "200", folder=figs_folder)
		# buf = make_movie(model, f, -3.0, 3.0, "300", folder=figs_folder)

	#os.makedirs(results_dir, exist_ok=True)
	# with open(results_dir + '/results.yaml', 'w') as outfile:
	# 	e1, e2, e3, l1 = (float("{:.6E}".format(error1)), float("{:.6E}".format(error2)), 
	# 		float("{:.6E}".format(error3)), float("{:.6E}".format(loss_value)))
	# 	trainTime = "{:.2F} s".format(toc - tic)
	# 	yaml.dump({'error1': e1, 'error2': e2, 'error3': e3, 'loss_value': l1,
	# 	'training_time': trainTime}, outfile, default_flow_style=False)