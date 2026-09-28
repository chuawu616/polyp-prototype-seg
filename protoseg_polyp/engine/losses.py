"""Losses used across the project, combined by `Criterion` from the `loss:` section of a config."""
import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU (PraNet / EMCAD).

    Kept verbatim for reproducibility: `reduce='none'` is a deprecated argument that PyTorch maps to
    reduction='mean', so the BCE term is a plain mean and its edge weighting has no effect.
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(fg_prob, mask):
    inter = (fg_prob * mask).sum(dim=(2, 3))
    union = fg_prob.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    return (1 - (2 * inter + 1) / (union + 1)).mean()


def kl_balance_loss(sim, mask, fg_channels, bg_channels, scale=10.0):
    """Push the average prototype usage inside each class toward uniform (Stage 2 KL experiment)."""
    probs = F.softmax(sim * scale, dim=1)
    loss = 0.0
    for chans, m in ((fg_channels, mask), (bg_channels, 1.0 - mask)):
        if m.sum() > 1.0:
            p = (probs[:, chans] * m).sum(dim=(0, 2, 3)) / m.sum()
            p = p / (p.sum() + 1e-8)
            t = torch.full_like(p, 1.0 / len(chans))
            loss = loss + torch.sum(p * (torch.log(p + 1e-8) - torch.log(t + 1e-8)))
    return loss


def orthogonality_loss(easy, hard):
    sim = F.normalize(easy, p=2, dim=1) @ F.normalize(hard, p=2, dim=1).t()
    return torch.mean(sim ** 2)


def ppc_loss(logits, target, ignore_index=-1):
    """Pixel-prototype contrast: cross-entropy over all K*M sub-prototypes."""
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)


def ppd_loss(logits, target, temperature=None, legacy_scaled=False, ignore_index=-1):
    """Pixel-prototype distance: (1 - cos(pixel, assigned prototype))^2.

    V0/V3 feed logits already multiplied by a temperature. The original code applied PPD to those
    scaled logits (legacy_scaled=True), which minimises (1 - T*cos)^2, i.e. pulls cos toward 1/T.
    By default the temperature is divided out so PPD acts on the cosine as in ProtoSeg.
    """
    keep = target != ignore_index
    logits, target = logits[keep], target[keep]
    if temperature is not None and not legacy_scaled:
        logits = logits / temperature
    sel = torch.gather(logits, 1, target[:, None].long())
    return (1 - sel).pow(2).mean()


def pseudo_ce_loss(multiclass, pseudo_labels, pseudo_hw):
    """K*M-way CE between the student logits and nearest-upsampled Sinkhorn pseudo-labels."""
    b = multiclass.shape[0]
    target = pseudo_labels.view(b, 1, *pseudo_hw).float()
    target = F.interpolate(target, size=multiclass.shape[-2:], mode='nearest').squeeze(1).long()
    return F.cross_entropy(multiclass, target)


class Criterion:
    """Sum of the losses named in the config, e.g. {'structure': 1.0, 'kl': {'weight': 0.1, 'ramp': [5, 10]}}."""

    def __init__(self, cfg, head):
        self.cfg, self.head = cfg, head

    @staticmethod
    def _weight(spec, epoch):
        if not isinstance(spec, dict):
            return float(spec)
        w = float(spec['weight'])
        if 'ramp' in spec:                 # 0 until ramp[0], linear to w at ramp[1]
            lo, hi = spec['ramp']
            w = 0.0 if epoch <= lo else w * min(1.0, (epoch - lo) / (hi - lo))
        return w

    def __call__(self, out, mask, epoch):
        terms = {}
        c = self.cfg
        if 'structure' in c:
            terms['structure'] = self._weight(c['structure'], epoch) * structure_loss(out['fg_logit'], mask)
        if 'seg_ce' in c:
            terms['seg_ce'] = self._weight(c['seg_ce'], epoch) * F.cross_entropy(out['scores'], mask.squeeze(1).long())
        if 'dice' in c:
            terms['dice'] = self._weight(c['dice'], epoch) * dice_loss(out['fg_prob'], mask)
        if 'kl' in c:
            w = self._weight(c['kl'], epoch)
            terms['kl'] = w * kl_balance_loss(out['sim'], mask, self.head.fg_channels, self.head.bg_channels)
        if 'orth' in c:
            terms['orth'] = self._weight(c['orth'], epoch) * orthogonality_loss(*self.head.prototype_groups())
        if 'ppc' in c and 'contrast_logits' in out:
            logits = out.get('contrast_logits_node', out['contrast_logits'])
            target = out.get('contrast_target_node', out['contrast_target'])
            terms['ppc'] = self._weight(c['ppc'], epoch) * ppc_loss(logits, target)
        if 'ppd' in c and 'contrast_logits' in out:
            spec = c['ppd']
            legacy = isinstance(spec, dict) and spec.get('legacy_scaled', False)
            terms['ppd'] = self._weight(spec, epoch) * ppd_loss(
                out['contrast_logits'], out['contrast_target'], out.get('temperature'), legacy)
        if 'pseudo_ce' in c and 'pseudo_labels' in out:
            terms['pseudo_ce'] = self._weight(c['pseudo_ce'], epoch) * pseudo_ce_loss(
                out['multiclass'], out['pseudo_labels'], out['pseudo_hw'])
        total = sum(terms.values())
        return total, {k: float(v) for k, v in terms.items()}
