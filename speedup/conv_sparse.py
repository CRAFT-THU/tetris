#!/usr/bin/env python

from blocksparse.conv import BlocksparseConv

import tensorflow as tf
from tensorflow.python.client import timeline
import numpy
import json

import sys
import os
import random
from collections import namedtuple

# configure tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def profile(bs, iw, ih, ic, oc, kw, kh, bi, bo, sp):
	if bi > ic:
		bi = ic
	if bo > oc:
		bo = oc

	num_input_blocks = ic / bi
    num_output_blocks = oc / bo
    num_blocks = num_input_blocks * num_output_blocks
    num_pruned_blocks = int(num_blocks * sp)
    num_remain_blocks = num_blocks - num_pruned_blocks
    actual_sparsity = num_pruned_blocks / float(num_blocks)
    
    # generate layout
    layout = np.array([0] * num_pruned_blocks + [1] * num_remain_blocks)
    np.random.shuffle(layout)
    layout = layout.reshape((num_input_blocks, num_output_blocks))

    # generate BCK according to layout
    # BCK is a list of blocks, each block is a tuple of two list: row indices and column indices
    BCK = []
    for i in range(num_input_blocks):
	for j in range(num_output_blocks):
	    if layout[i, j] == 1:
		BCK.append((
		    [c for c in range(i * bi, (i + 1) * bi)],
		    [k for k in range(j * bo, (j + 1) * bo)]
		))
    TRS = (kw, kh)
    DHW = (iw, ih)


    # generate random shuffle order
    indices = range(oc)
    random.shuffle(indices)
    
    tf.reset_default_graph()
    with tf.Session() as sess:
    	# generate operation
    	bs_conv = BlocksparseConv(BCK, TRS, DHW)

    	# build computational graph
    	x = tf.placeholder(tf.float32, shape=bs_conv.i_shape(bs))
    	k = tf.get_variable("k", shape=bs_conv.f_shape(), dtype=tf.float32)
    	i = tf.constant(indices)
    	y = bs_conv(k, x)
    	y = tf.gather(y, i, axis=1)
    
    	# run and profile
    	options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    	run_metadata = tf.RunMetadata()
    	sess.run(tf.global_variables_initializer())
    	sess.run(y, feed_dict={x: np.ones(shape=bs_conv.i_shape(bs), dtype='float32')}, options=options, run_metadata=run_metadata)
    	fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    	chrome_trace = fetched_timeline.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
		f.write(chrome_trace)

    # parse the trace
    with open('timeline.json', 'r') as f:
		o = json.load(f)['traceEvents']
    conv_time = int(next(item for item in o if item['name'] == u'BlocksparseConv')['dur'])
    gather_time = int(next(item for item in o if item['name'].startswith(u'Gather'))['dur'])

    os.remove('timeline.json')

    return actual_sparsity, conv_time + gather_time

vgg16_config = {
    "conv1.1": [224, 224, 3, 64, 3, 3],
    "conv1.2": [224, 224, 64, 64, 3, 3],
    "conv2.1": [112, 112, 64, 128, 3, 3],
    "conv2.2": [112, 112, 128, 128, 3, 3],
    "conv3.1": [56, 56, 128, 256, 3, 3],
    "conv3.2": [56, 56, 256, 256, 3, 3],
    "conv3.3": [56, 56, 256, 256, 3, 3],
    "conv4.1": [28, 28, 256, 512, 3, 3],
    "conv4.2": [28, 28, 512, 512, 3, 3],
    "conv4.3": [28, 28, 512, 512, 3, 3],
    "conv5.1": [14, 14, 512, 512, 3, 3],
    "conv5.2": [14, 14, 512, 512, 3, 3],
    "conv5.3": [14, 14, 512, 512, 3, 3]
}

vgg16_layers = [
	"conv1.1","conv1.2",
	"conv2.1","conv2.2",
	"conv3.1","conv3.2","conv3.3",
	"conv4.1","conv4.2","conv4.3",
	"conv5.1","conv5.2","conv5.3"
]

def main():
	if len(sys.argv) != 2:
		print("Please specify the configuration file which define the block size and pruning rate.")
		sys.exit()

	config_fn = sys.argv[1]
	Config = namedtuple('Config', ['block_sizes', 'pruning_rates'])
	with open(config_fn, 'r') as f:
	    configuration = json.load(f)
	configuration = Config(**configuration['vgg16_bn'])

	with open("conv_sparse.csv", "w") as f:
		#batch size is 64
		bs = 64

		f.write("layer, sparsity, time\n")
		for k, (iw, ih, ic, oc, kw, kh) in vgg16_config:
			i = vgg16_layers.index(k)
			b = configuration.block_sizes[i]
			bi = b[0] if b[0] > 0 else ic
			bo = b[1] if b[1] > 0 else oc
			sp = configuration.pruning_rates[i]
			sparsity, exec_time = profile(bs, iw, ih, ic, oc, kw, kh, bi, bo, sp)
			exec_time *= 1e-6
			f.write("%s, %f, %f\n" % (k, sparsity, exec_time))

if __name__ == "__main__":
    main()