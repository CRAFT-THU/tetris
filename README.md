# Tetris
## Introduction
This project includes the source code of NIPS 2018 paper:

    @incollection{NIPS2018_7666,
      title = {TETRIS: TilE-matching the TRemendous Irregular Sparsity},
      author = {Ji, Yu and Liang, Ling and Deng, Lei and Zhang, Youyang and Zhang, Youhui and Xie, Yuan},
      booktitle = {Advances in Neural Information Processing Systems 31},
      editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
      pages = {4119--4129},
      year = {2018},
      publisher = {Curran Associates, Inc.},
      url = {http://papers.nips.cc/paper/7666-tetris-tile-matching-the-tremendous-irregular-sparsity.pdf}
    }


Please cite this paper if you use the code.


## Documentation

### Speedup
The 'speedup' folder contains code and result for speedup.
To run this code, make sure tensorflow, pytorch and [blocksparse](https://github.com/openai/blocksparse) (from OpenAI) are installed.

We have sparse and dense result for convolution and matrix multiplication.
The sparse cases are based on blocksparse library with tensorflow.
The baseline cases are based on pytorch because tensorflow does not support convolution without cudnn, and we plan to use cublas as baseline.

To get the results, just run the four python code in [speedup folder](speedup/).
For [conv_sparse.py](speedup/conv_sparse.py), you should specify the configuration file (see [example](accuracy/config.json.example)) as the command line argument.

### Accuracy
The 'accuracy' folder contains code and configurations for accuracy.
To run this code, make sure pytorch, torchvision are installed and run the following command to prune and fine-tune the model.
```bash
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py /path/to/ImageNet/ "config.json" --arch=VGG16_bn --lr=1e-3 --batch-size=64 --prefix=VGG16_bn
```
You should specify the path to ImageNet dataset, the model architecture used (in torchvision), and the pruning configurations (the json file, see [example](accuracy/config.json.example)).

We have provide many configuration files (all the json files).
It provide the block size for each dimension of each layer and the pruning rates.

### Important notes
Pruning codes are implemented on GPU.
It takes about 1 minute and 12GB GPU memory to prune VGG16.
Make sure your GPU device provides enough memory.
