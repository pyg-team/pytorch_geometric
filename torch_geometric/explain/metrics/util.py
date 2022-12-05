import torch


def hard_threshold(mask_dict, threshold):
    """Impose hard threshold on a dictionary of masks"""
    mask_dict = {
        key: (mask > threshold).float()
        for key, mask in mask_dict.items()
    }
    return mask_dict


def topk_threshold(mask_dict, threshold, hard=False):
    """Impose topk threshold on a dictionary of masks"""
    for key, mask in mask_dict.items():
        if threshold >= mask.numel():
            if hard:
                mask_dict[key] = torch.ones_like(mask)
            continue

        value, index = torch.topk(
            mask.flatten(),
            k=threshold,
        )

        out = torch.zeros_like(mask.flatten())
        if not hard:
            out[index] = value
        else:
            out[index] = 1.0
        mask_dict[key] = out.reshape(mask.size())

    return mask_dict
