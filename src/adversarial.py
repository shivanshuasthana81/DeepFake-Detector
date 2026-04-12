import torch
import torch.nn.functional as F


def fgsm_attack(feat_extractor, model, seqs, labels, epsilon):
    seqs.requires_grad = True

    B, S, C, H, W = seqs.shape
    seqs_flat = seqs.view(B*S, C, H, W)

    feats = feat_extractor(seqs_flat)
    feats = feats.view(B, S, -1)

    logits, _ = model(feats)
    loss = F.cross_entropy(logits, labels)

    feat_extractor.zero_grad()
    model.zero_grad()
    loss.backward()

    grad = seqs.grad.data
    perturbed = seqs + epsilon * grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed.detach()


def pgd_attack(feat_extractor, model, seqs, labels,
               epsilon=0.03, alpha=0.01, iters=5):

    original = seqs.clone().detach()

    for _ in range(iters):
        seqs.requires_grad = True

        B, S, C, H, W = seqs.shape
        seqs_flat = seqs.view(B*S, C, H, W)

        feats = feat_extractor(seqs_flat)
        feats = feats.view(B, S, -1)

        logits, _ = model(feats)
        loss = F.cross_entropy(logits, labels)

        feat_extractor.zero_grad()
        model.zero_grad()
        loss.backward()

        grad = seqs.grad.data
        seqs = seqs + alpha * grad.sign()

        eta = torch.clamp(seqs - original, min=-epsilon, max=epsilon)
        seqs = torch.clamp(original + eta, 0, 1).detach()

    return seqs
