import torch
import torch.nn as nn 

# https://github.com/facebookresearch/mae/blob/main/models_mae.py

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    # print("noise:",noise)
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    print("ids_shuffle:",ids_shuffle)
    print("ids_restore:",ids_restore)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    #这个不是应该保留的？非原始顺序，没问题？
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    print("ids_keep:",ids_keep)
    print("x_masked:",x_masked) 

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    print("mask:",mask)

    return x_masked, mask, ids_restore

def unmask(x, ids_restore):  
    # append mask tokens to sequence
    mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    print("mask_token:",mask_token)
    
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    print("mask_tokens:",mask_tokens)
    
    # x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    x_ = torch.cat([x[:, :, :], mask_tokens], dim=1) 
    print("x_ 1:", x_)
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    print("x_ 2:", x_)
    # x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    # print("x_ 3:", x_)

    return x

batch, length, embed_dim = 2,5,3
input = torch.rand(batch, length, embed_dim)

print(input.shape)
print("x:",input)

x_masked, mask, ids_restore = random_masking(input,0.5)

print("--------------")

unmask(x_masked, ids_restore)