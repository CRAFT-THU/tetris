#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.autograd.Variable as Variable
import sys
import time

# cudnn is tooooo fast!! it has too many in house optimization. we use cublas as baseline
torch.backends.cudnn.enabled = False

# run for 100 times
def profile(bs, iw, ih, ic, oc, kw, kh, times=100):
	conv = nn.Conv2d(ic, oc, kernel_size=(kw, kh), stride=(1, 1), padding=(kw/2, kh/2)).cuda()
	x = Variable(torch.randn(bs, ic, iw, ih)).cuda()
	# run once to prevent initialization overhead
	y = conv(x)

	#profile
	torch.cuda.synchronize()
	begin = time.time()
	for i in range(times):
		y = conv(x)
	torch.cuda.synchronize()
	end = time.time()

	return (end - begin) / times

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

def main():
	# batch size
	bs = 64

	with open("conv_dense.csv", "w") as f:
		f.write("layer, time\n")
		for k, (iw, ih, ic, oc, kw, kh) in vgg16_config:
			f.write("%s, %f\n" % (k, profile(bs, iw, ih, ic, oc, kw, kh)))

if __name__ == "__main__":
	main()