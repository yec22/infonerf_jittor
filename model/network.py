import jittor as jt
from jittor import nn, Module

# Positional encoding
class Embedder:
    def __init__(self, freq):
        self.freq = freq
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []

        d = 3
        embed_fns.append(lambda x : x)
        out_dim = d     
        max_freq = self.freq - 1
        N_freqs = self.freq
        
        freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)          
        for freq in freq_bands:
            for p_fn in [jt.sin, jt.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
        
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return jt.concat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(freq): 
    embedder_obj = Embedder(freq)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# NeRF Model, Reference: https://github.com/bmild/nerf
class NeRF(Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, skips=[4]):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)

    def execute(self, x):
        input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = nn.relu(h)
            if i in self.skips:
                h = jt.concat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = jt.concat([feature, input_views], -1)
        
        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = nn.relu(h)

        rgb = self.rgb_linear(h)
        outputs = jt.concat([rgb, alpha], -1)

        return outputs # (rgb, alpha) -> (3, 1)
