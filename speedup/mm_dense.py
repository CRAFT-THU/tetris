from blocksparse.matmul import BlocksparseMatMul

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np

import os
import json

batch_size = 256
input_size = 4096
output_size = 4096

def main():
	x = tf.placeholder(tf.float32, shape=[batch_size, input_size])
	w = tf.get_variable("w", [input_size, output_size], dtype=tf.float32)
	y = tf.matmul(x, w)

	sess = tf.Session()
	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	sess.run(tf.global_variables_initializer())
	sess.run([y], feed_dict = {x: np.ones((batch_size, input_size), dtype='float32')}, options=options, run_metadata=run_metadata)
	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
	with open('timeline.json', 'w') as f:
	    f.write(chrome_trace)

	with open('timeline.json', 'r') as f:
	    o = json.load(f)['traceEvents']

	exec_time = int(next(item for item in o if item['name'] == u'MatMul')['dur'])
	exec_time *= 1e-6
	os.remove('timeline.json')
	with open("mm_dense.csv", "w") as f:
		f.write("input_size, output_size, time\n")
		f.write("%d, %d, %f\n" % (input_size, output_size, exec_time))

if __name__ == '__main__':
	main()