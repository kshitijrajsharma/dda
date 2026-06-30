import torch

from dda.losses import IGNORE_INDEX, LocalizationLoss, OrdinalDamageLoss


def _pixel_logits(pred_class: int, confidence: float = 6.0) -> torch.Tensor:
    logits = torch.zeros(1, 4, 1, 1)
    logits[0, pred_class, 0, 0] = confidence
    return logits


def test_ordinal_penalises_far_errors_more():
    """Predicting 'destroyed' when truth is 'no-damage' must cost more than predicting 'minor'."""
    loss = OrdinalDamageLoss()
    target = torch.zeros(1, 1, 1, dtype=torch.long)
    near = loss(_pixel_logits(1), target)
    far = loss(_pixel_logits(3), target)
    assert far > near


def test_background_only_crop_is_zero_not_nan():
    loss = OrdinalDamageLoss()
    target = torch.full((1, 1, 1), IGNORE_INDEX, dtype=torch.long)
    value = loss(_pixel_logits(2), target)
    assert torch.isfinite(value)
    assert float(value) == 0.0


def test_class_weights_shift_loss():
    # Two pixels: class 0 predicted wrong (high loss), class 1 predicted right (low loss).
    # Upweighting class 0 must raise the weighted-mean loss.
    logits = torch.zeros(1, 4, 1, 2)
    logits[0, 1, 0, 0] = 6.0  # pixel 0 (true class 0) confidently predicts class 1 -> wrong
    logits[0, 1, 0, 1] = 6.0  # pixel 1 (true class 1) confidently predicts class 1 -> right
    target = torch.tensor([[[0, 1]]])
    plain = OrdinalDamageLoss()(logits, target)
    weighted = OrdinalDamageLoss(class_weights=torch.tensor([5.0, 1.0, 1.0, 1.0]))(logits, target)
    assert weighted > plain


def test_localization_loss_finite():
    logit = torch.randn(2, 16, 16)
    target = (torch.rand(2, 16, 16) > 0.5).float()
    assert torch.isfinite(LocalizationLoss()(logit, target))
