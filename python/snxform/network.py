"""Helper functions and classes to implement the NN transformer.
"""

import numpy as np
import json
import torch
import torch.nn as nn


#- Helper methods for ViT
def patchify(spectra: torch.Tensor, n_patches: int) -> torch.Tensor:
    """
		Parameters
	  ----------
    spectra: 1D spectra: torch.Tensor of shape (N, 1, len_spectrum)
    n_patches: number of patches to break the spectra into (must be a factor of len_spectrum)

    Returns
    -------
    patches of the spectra: torch.Tensor of shape (N, n_patches, len_spectrum // n_patches)
    """
    
    n, _, l_spectrum = spectra.shape
    
    # create patches
    patch_size = l_spectrum // n_patches
    patches = torch.zeros(n, n_patches, l_spectrum // n_patches)
    for idx, spectrum in enumerate(spectra):
        for i in range(n_patches):
            patch = spectrum[:, i * patch_size: (i + 1) * patch_size]
            patches[idx, i] = patch
    
    return patches


def positional_embedding(i: int, j: int, d: int) -> float:
    """
    Parameters
    ----------
    i: tensor index
    j: embedding index
    d: embedding dimension

    Returns
    -------
    positional embedding for i, j
    """
    
    if j % 2 == 0:
        return np.sin(i / (10000 ** (j / d)))
    return np.cos(i / (10000 ** ((j - 1) / d)))


def get_positional_embeddings(sequence_length: int, d: int) -> torch.Tensor:
    """
    Parameters
    ----------
    sequence_length: length of sequence
    d: embedding dimension

    Returns
    -------
    positional embeddings for sequence of length sequence_length
    """
    
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = positional_embedding(i, j, d)
    
    return result

class MSA(nn.Module):
    """
    Multi-Head Self-Attention
    """
    
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads
        
        assert d % n_heads == 0, f"Cannot divide dimension {d} into {n_heads} heads"
        
        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, sequences):
        """
        Sequences have shapes (N, seq_length, token_dim)
        We must transform to shape (N, seq_length, n_heads, token_dim / n_heads)
        and concatenate back into (N, seq_length, token_dim)
        """
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                
                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                
                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class ViTBlock(nn.Module):
    """
    Transformer Encoder Block
    """
    
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(nn.Linear(hidden_d, mlp_ratio * hidden_d), nn.GELU(),
                                 nn.Linear(mlp_ratio * hidden_d, hidden_d))
    
    def forward(self, x):
        """
        Encoder1 will normalize input, pass through MSA,
        add residual connection

        Encoder2 will normalize encoder1, pass through MLP
        """
        encoder1 = x + self.mhsa(self.norm1(x))
        encoder2 = encoder1 + self.mlp(self.norm2(encoder1))
        return encoder2


class ViT(nn.Module):
    def __init__(self, cl=(1, 1024), n_patches=64, n_blocks=2, hidden_d=8, n_heads=2, out_d=10, device = None):
        super(ViT, self).__init__()
        
        # If device is provided, use that. 
        # HOwever, if CUDA is availible, that is the default device
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ViT IS NOW IN {self.device}")
        
        self.cl = cl  # (channels, length)
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        assert cl[1] % n_patches == 0, "Image length must be divisible by n_patches"
        self.patch_size = cl[1] // n_patches
        
        # Linear mapping of patches to hidden dimension
        self.input_d = int(cl[0] * self.patch_size)
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d).to(self.device)
        
        # Classification Token
        #self.class_token = nn.Parameter(torch.rand(1, self.hidden_d).to(self.device))
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d)).to(self.device)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(get_positional_embeddings(
            n_patches + 1, self.hidden_d).clone().detach())
        self.pos_embed.requires_grad = False
        
        # Transformer Encoder
        self.blocks = nn.ModuleList(
            [ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]).to(self.device)
        
        # Classification mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1)).to(self.device)

    
    def forward(self, images):
        n, _, _ = images.shape
        patches = patchify(images, self.n_patches).to(self.device)
        
        # Linear tokenization --> map vector corresponding to each patch to hidden dimension
        image_tokens = self.linear_mapper(patches)
        
        # Adding classification
        tokens = torch.stack([torch.vstack(
            (self.class_token, image_tokens[i])) for i in range(len(image_tokens))])
        
        # Adding positional embeddings
        pos_embed = self.pos_embed.repeat(n, 1, 1)
        out = tokens + pos_embed
        
        for block in self.blocks:
            out = block(out)
        
        # For classification, we take the first token
        out = out[:, 0]
        
        return self.mlp(out)
        
    
    def saveparams(self, model_name):
        dict = {'cl': self.cl, 'patches':self.n_patches, 'n_blocks':self.n_blocks, 'n_heads':self.n_heads, 'hidden_d':self.hidden_d}
        with open(f'{model_name}_parameters.json', 'w') as f:
            json.dump(dict, f)
            
def loadparams(model_name):
    with open(f'{model_name}_parameters.json', 'r') as f:
        params = json.load(f)
    return params
