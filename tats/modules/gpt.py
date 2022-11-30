
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

class GPTConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked") and config.n_unmasked > 0:
            mask[:, :config.n_unmasked+1] = 1
            mask[:, -config.n_unmasked+1:] = 1
            mask[-config.n_unmasked+1:, config.n_unmasked+1:-config.n_unmasked+1] = 0
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
            
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_drop(self.proj(y))
        return y, present

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(), 
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        if return_present: assert not self.training
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x

class GPT(nn.Module):
    def __init__(self, args, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0, vtokens_pos=False):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.vtokens_pos = vtokens_pos
        if self.vtokens_pos:
            self.vtokens_pos_emb = nn.Parameter(torch.zeros(1, args.sequence_length, args.resolution, args.resolution, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None, cbox=None, tbox=None):
        token_embeddings = self.tok_emb(idx)

        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]
        if self.vtokens_pos:
            if tbox:
                vtokens_position_embeddings = torch.cat([self.vtokens_pos_emb[:, tpos[0]:tpos[1], pos[0]:pos[1], pos[2]:pos[3], :].reshape(1, -1, self.vtokens_pos_emb.shape[-1]) for pos, tpos in zip(cbox, tbox)], 0)
            else:
                vtokens_position_embeddings = torch.cat([self.vtokens_pos_emb[:, :, pos[0]:pos[1], pos[2]:pos[3], :].reshape(1, -1, self.vtokens_pos_emb.shape[-1]) for pos in cbox], 0)
            position_embeddings = position_embeddings + vtokens_position_embeddings
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None, cbox=None):
        assert not self.training
        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]
            if self.vtokens_pos:
                vtokens_position_embeddings = torch.cat([self.vtokens_pos_emb[:, :, pos[0]:pos[1], pos[2]:pos[3], :].reshape(1, -1, self.vtokens_pos_emb.shape[-1]) for pos in cbox], 0)
                vtokens_position_embeddings = vtokens_position_embeddings[:, past_length, :]
                position_embeddings = position_embeddings + vtokens_position_embeddings
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]
            if self.vtokens_pos:
                vtokens_position_embeddings = torch.cat([self.vtokens_pos_emb[:, :, pos[0]:pos[1], pos[2]:pos[3], :].reshape(1, -1, self.vtokens_pos_emb.shape[-1]) for pos in cbox], 0)
                vtokens_position_embeddings = vtokens_position_embeddings[:, :token_embeddings.shape[1], :]
                position_embeddings = position_embeddings + vtokens_position_embeddings

        x = self.drop(token_embeddings + position_embeddings)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)

    @torch.no_grad()
    def forward_with_past_and_future(self, idx, idx_future=None, embeddings=None, targets=None, past=None, past_length=None, future_length=None):
        assert not self.training
        if past is None:
            token_embeddings_past = self.tok_emb(idx)
            token_embeddings_future = self.tok_emb(idx_future)
            token_embeddings = torch.cat([token_embeddings_past, token_embeddings_future], dim=1)
        else:
            token_embeddings = self.tok_emb(idx)

        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None and future_length is not None
            past = torch.cat(past, dim=-2)
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, idx.shape[0], self.config.n_head, past_length+future_length, self.config.n_embd//self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]
        else:
            position_embeddings_past = self.pos_emb[:, :token_embeddings_past.shape[1], :]
            position_embeddings_future = self.pos_emb[:, -token_embeddings_future.shape[1]:, :]
            position_embeddings = torch.cat([position_embeddings_past, position_embeddings_future], dim=1)

        x = self.drop(token_embeddings + position_embeddings)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)
    
@torch.no_grad()
def sample_with_past(x, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None, cbox=None):
    sample = x
    cond_len = x.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        if cbox is None:
            logits, _, present = model.forward_with_past(x, past=past, past_length=(n+cond_len-1))
        else:
            logits, _, present = model.forward_with_past(x, past=past, past_length=(n+cond_len-1), cbox=cbox)
        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_len:]
    return sample

@torch.no_grad()
def sample_with_past_and_future(x, x_future, model, steps, temperature=1., sample_logits=True,
                     top_k=None, top_p=None, callback=None, cbox=None):
    sample = x
    cond_len = x.shape[1]
    future_length = x_future.shape[1]
    past = None
    for n in range(steps):
        if callback is not None:
            callback(n)
        logits, _, present = model.forward_with_past_and_future(x, idx_future=x_future, past=past, past_length=(n+cond_len-1), future_length=future_length)
        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)
        sample = torch.cat((sample, x), dim=1)
    del past
    sample = sample[:, cond_len:]
    return sample
