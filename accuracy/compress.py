from __future__ import print_function

import itertools
import torch
import numpy as np

__all__ = ['blocksparse']

def blocksparse(X, block_sizes, pruning_rate, shuffle=True):
    """Blocksparse pruning with pytorch on GPU
    
    For input tensor X, we use an EM algorithm to determine the best shuffling order.
        E-step: measure the importance of each block and mark the least important blocks to be pruned.
        M-step: shuffle different dimensions of X to minimize the elements inside pruned blocks.

    The M-step is not that easy, we also use an iterative algorithm to minimize the elements.
    We fix other dimensions and shuffle one dimension each time.

    suppose the mask is M, we have S = XM^T where S_ij represents the pruned value if we put the i-th row to j-th mask. Thus S + S^T describe the pruned value when we swap i and j. The original pruned value in the two columns are tr(S) + tr(S)^T.

    Thus, we find the maximum value of (tr(S) + tr(S)^T) - (S + S^T) and swap i j if it is larger than zero.

    Args:
        x (np.ndarray): a tensor to be pruned.
        block_sizes (tuple[int]): block sizes for each dimension of 'x'.
        pruning_rate (float): pruning rate.
        shuffle (bool): Shuffle or not.

    Returns:
        (tuple[tuple[int]], np.ndarray): shuffle orders and mask
    """
    ## prepare
    X = torch.abs(X).cuda()

    dim_sizes = X.size() 
    num_dims = len(dim_sizes)
    block_sizes = [bs if bs > 0 else ds for bs, ds in zip(block_sizes, dim_sizes)] 
    block_nums = [int((ds - 1) / bs) + 1 for bs, ds in zip(block_sizes, dim_sizes)]
    orders = [torch.arange(ds).long() for ds in dim_sizes]
    num_blocks = np.prod(block_nums)
    num_pruned_blocks = int(num_blocks * pruning_rate)

    ## EM iteration
    print("=> begin EM iteration: block_sizes %s, pruning_rate %f" % (str(block_sizes), pruning_rate))
    while True:
        ## E step: choose block to be pruned
        # compute sum of each block
        block_sums = X.view(*tuple(itertools.chain.from_iterable((bn, bs) for bn, bs in zip(block_nums, block_sizes))))
        for i in range(num_dims):
            block_sums = block_sums.sum(i + 1)
        # choose the blocks to be pruned
        block_mask = torch.zeros_like(block_sums)
        if num_pruned_blocks > 0:
            block_mask[np.unravel_index(block_sums.view(-1).sort()[1][:num_pruned_blocks], dims=block_nums)] = 1
        mask = (block_mask[tuple([slice(None), None] * num_dims)] * torch.ones(*block_sizes).cuda()[tuple([None, slice(None)] * num_dims)]).view(*dim_sizes)

        prev_pruned_sum = (X * mask).sum()
        print("==> E-step: pruned sum is %f" % prev_pruned_sum)

        if not shuffle:
            return orders, 1 - mask

        ## M step: determine the best order
        for axis in range(num_dims):
            if dim_sizes[axis] == block_sizes[axis]:
                print("Skip axis %d" % axis)
                continue
            order = torch.arange(dim_sizes[axis]).long()
            S = torch.mm(X.transpose(0, axis).contiguous().view(X.size(axis), -1), mask.transpose(0, axis).contiguous().view(mask.size(axis), -1).t())
            D = torch.diagonal(S)
            G = D[:, None] + D[None, :]
            G -= S
            G -= S.t()
            G_maxes_v, G_maxes_i = torch.max(G, -1)
            G_max_v, G_max_i = torch.max(G_maxes_v, -1)
            i, j = G_max_i, G_maxes_i[G_max_i]
            while G_max_v >= 1e-3:
                ## swap i, j
                S[[i,j], :] = S[[j,i], :]
                order[[i, j],] = order[[j, i],]
                print("====> Swap gain %f, (%d, %d)" % (G_max_v, i, j), end="\r")
		
                # update D and G
                D = torch.diagonal(S)
                G_i = D + D[i] - S[i, :] - S[:, i]
                G_j = D + D[j] - S[j, :] - S[:, j]
                G[i, :] = G_i
                G[:, i] = G_i
                G[j, :] = G_j
                G[:, j] = G_j

                # update G_max
                indices = torch.cat(((G_maxes_i == G_maxes_i[i]).nonzero(), (G_maxes_i == G_maxes_i[j]).nonzero())).view(-1)
                if indices.size(0) > dim_sizes[axis] / 2:
                    G_maxes_v, G_maxes_i = torch.max(G, -1)
                    G_max_v, G_max_i = torch.max(G_maxes_v, -1)
                    i, j = G_max_i, G_maxes_i[G_max_i]
                    continue

                G_maxes_v[indices], G_maxes_i[indices] = torch.max(G[indices, :], -1)
                indices = (G_maxes_v < G[:, i])
                G_maxes_v[indices] = G[indices, i]
                G_maxes_i[indices] = i
                indices = (G_maxes_v < G[:, j])
                G_maxes_v[indices] = G[indices, j]
                G_maxes_i[indices] = j
                G_max_v, G_max_i = torch.max(G_maxes_v, -1)
                i, j = G_max_i, G_maxes_i[G_max_i]
            
            orders[axis] = orders[axis][order]
            X = X[tuple(order if k == axis else slice(None) for k in range(num_dims))]
            print("===> axis %d, pruned sum is %f" % (axis, (X * mask).sum()))

        pruned_sum = (X * mask).sum()
        print("==> M-step: pruned sum is %f" % pruned_sum)
        if prev_pruned_sum - pruned_sum < 1e-3:
            break
        else:
            prev_pruned_sum = pruned_sum

    ## generate reverse mask
    for axis in range(num_dims):
        mask[tuple(orders[dim] if dim == axis else slice(None) for dim in range(num_dims))] = mask.clone()
    mask = mask.cpu()
    torch.cuda.empty_cache()
    return orders, 1 - mask
