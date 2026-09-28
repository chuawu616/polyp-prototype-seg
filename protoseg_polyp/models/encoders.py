"""Feature encoders. Each returns a dict of feature maps; `out_key` names the map the head uses."""
import torch
import torch.nn as nn

from .pvtv2 import pvt_v2_b2

try:  # EMCAD may not be redistributed; tools/fetch_third_party.py downloads it from the official repository
    from .third_party.emcad_decoders import EMCAD
except ImportError as e:  # pragma: no cover
    raise ImportError('EMCAD decoder not found - run `python tools/fetch_third_party.py` once.') from e


class PVTEMCAD(nn.Module):
    """PVTv2-b2 encoder + EMCAD decoder, the TA-provided baseline.

    Feature nodes at 352x352 input:
        x1..x4  : encoder stages                           (64/128/320/512 ch, stride 4/8/16/32)
        dd4..dd1: decoder stage inputs; dd4 = x4, dd3..dd1 = sum of upsampled path and gated skip,
                  i.e. the inputs of the decoder's cab3 / cab2 / cab1 blocks   (512/320/128/64 ch)
        d1      : decoder output (64 ch, stride 4, 88x88)  <- used by every head

    The official EMCAD decoder is used unmodified (expansion_factor=2, as in the TA's code); the intermediate
    nodes are captured with forward pre-hooks. They are detached unless detach_taps=False (non-learnable V3
    back-propagates a loss through dd4).
    """
    out_key = 'd1'
    out_dim = 64

    def __init__(self, pretrained=None, detach_taps=True):
        super().__init__()
        self.detach_taps = detach_taps
        self.backbone = pvt_v2_b2()
        if pretrained:
            state = torch.load(pretrained, map_location='cpu', weights_only=True)
            model_dict = self.backbone.state_dict()
            model_dict.update({k: v for k, v in state.items() if k in model_dict})
            self.backbone.load_state_dict(model_dict)
        self.decoder = EMCAD(channels=[512, 320, 128, 64], expansion_factor=2)
        self._taps = {}
        for name in ('dd3', 'dd2', 'dd1'):
            block = getattr(self.decoder, 'cab' + name[-1])
            block.register_forward_pre_hook(lambda m, inp, name=name: self._taps.__setitem__(name, inp[0]))

    def _tap(self, t):
        return t.clone().detach() if self.detach_taps else t.clone()

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        d1 = self.decoder(x4, [x3, x2, x1])[-1]
        taps = {k: self._tap(v) for k, v in self._taps.items()}
        self._taps.clear()
        return dict(x1=x1, x2=x2, x3=x3, x4=x4, dd4=self._tap(x4), d1=d1, **taps)


DINOV3_DIMS = {'vits16': 384, 'vits16plus': 384, 'vitb16': 768, 'vitl16': 1024}


class DINOv3(nn.Module):
    """DINOv3 ViT loaded through torch.hub; features from one or several blocks, concatenated.

    layers: int n -> last n blocks (original single-layer models use 1);
            tuple  -> 0-based block indices, e.g. (1, 3, 5, 7, 9, 11) = blocks 2, 4, ..., 12.
    Output stride is 16 (22x22 at 352x352).
    """
    out_key = 'feat'

    def __init__(self, arch='vits16plus', weights=None, layers=1, freeze=False,
                 hub_repo='facebookresearch/dinov3', hub_source='github'):
        super().__init__()
        self.freeze = freeze
        self.layers = tuple(layers) if isinstance(layers, (list, tuple)) else layers
        self.backbone = torch.hub.load(hub_repo, f'dinov3_{arch}', source=hub_source, pretrained=False)
        if weights:
            self.backbone.load_state_dict(torch.load(weights, map_location='cpu'), strict=True)
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        n_layers = len(self.layers) if isinstance(self.layers, tuple) else self.layers
        self.out_dim = DINOV3_DIMS[arch] * n_layers

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze):
            feats = self.backbone.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True)
        return dict(feat=torch.cat(feats, dim=1))
