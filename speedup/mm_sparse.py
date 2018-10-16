from blocksparse.matmul import BlocksparseMatMul

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np

import random
import os
import json

def profile(batch_size, input_size, output_size, block_size, sparsity):
    num_input_blocks = input_size / block_size
    num_output_blocks = output_size / block_size
    num_blocks = num_input_blocks * num_output_blocks
    num_pruned_blocks = int(num_blocks * sparsity)
    num_remain_blocks = num_blocks - num_pruned_blocks

    actual_sparsity = num_pruned_blocks / float(num_blocks)

    # generate layout
    layout = np.array([0] * num_pruned_blocks + [1] * num_remain_blocks)
    np.random.shuffle(layout)
    layout = layout.reshape((num_input_blocks, num_output_blocks))

    # generate shuffle order
    indices = range(output_size)
    random.shuffle(indices)
    
    tf.reset_default_graph()
    with tf.Session() as sess:
	bsmm = BlocksparseMatMul(layout, block_size=block_size)
	i = tf.constant(indices)
       	x = tf.placeholder(tf.float32, shape=(batch_size, input_size))
	w = tf.get_variable('w', bsmm.w_shape, dtype=tf.float32)
	y = bsmm(x, w)
	y = tf.gather(y, i, axis=1)

	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	run_metadata = tf.RunMetadata()
	sess.run(tf.global_variables_initializer())
	sess.run(y, feed_dict={x: np.ones((batch_size, input_size), dtype='float32')}, options=options, run_metadata=run_metadata)
	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
	chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
	f.write(chrome_trace)

    with open('timeline.json', 'r') as f:
	o = json.load(f)['traceEvents']
	mm_time = int(next(item for item in o if item['name'] == u'BlocksparseMatmul')['dur'])
	gather_time = int(next(item for item in o if item['name'].startswith(u'Gather'))['dur'])

    os.remove('timeline.json')

    return actual_sparsity, mm_time + gather_time

def main():
    with open('mm_sparse.csv', 'w') as f:
        f.write('block_size, sparsity, time\n')
        for block_size in (8, 16, 32):
        	for sparsity in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
     	    actual_sparsity, execution_time = profile(batch_size=256, input_size=4096, output_size=4096, block_size=block_size, sparsity=sparsity)
    	    f.write('%d, %f, %d\n' % (block_size, actual_sparsity, execution_time))

if __name__ == '__main__':
    main()