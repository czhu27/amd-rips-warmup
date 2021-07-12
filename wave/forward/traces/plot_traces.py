import numpy as np
import matplotlib.pyplot as plt

traces = np.load('traces.npz')['traces']

num_traces = traces.shape[0] - 1
for i in range(num_traces):
	fname = "trace{:02d}.png".format(i)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(traces[num_traces, :], traces[i, :])
	ax.set_xlabel("time")
	ax.set_ylabel("pressure")
	ax.set_title("Trace {:02d}".format(i))
	plt.tight_layout(pad=0.75)
	plt.savefig(fname)


