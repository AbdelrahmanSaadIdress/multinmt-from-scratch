"""
training/scheduler.py
======================
Warmup learning-rate schedule — Vaswani et al. (2017), Section 5.3.

Theory :
    The paper uses a custom schedule that increases the learning rate linearly
    for the first `warmup_steps` steps, then decays it proportionally to the
    inverse square root of the step number.  The warmup prevents the early
    training steps (when gradients are noisy and the model is random) from
    taking destabilisingly large steps that corrupt the attention weights.
    After warmup, the decay mirrors the inverse-sqrt schedule used in Adam's
    own adaptive learning rate, providing a complementary global scaling.

Paper equation (Section 5.3):
    lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5})

    This is equivalent to:
        if step < warmup_steps:   lrate = d_model^{-0.5} · step · warmup_steps^{-1.5}
        else:                     lrate = d_model^{-0.5} · step^{-0.5}

    The crossover point (where both expressions are equal) is exactly at
    step = warmup_steps.
"""

import logging
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def noam_lambda(step: int, d_model: int, warmup_steps: int) -> float:
    """
    Compute the learning-rate multiplier for step `step`.

    This is the pure function form of the Noam schedule.  PyTorch's
    LambdaLR multiplies the base optimizer lr by the returned value, so
    we set the optimizer's base lr to 1.0 and let this function carry all
    the scaling — otherwise we'd have a confusing interaction between the
    two.

    Parameters
    ----------
    step         : int   Current training step (1-indexed; we clamp to ≥1
                         to avoid division by zero on step 0).
    d_model      : int   Model dimension (determines the peak learning rate).
    warmup_steps : int   Number of linear warmup steps.

    Returns
    -------
    float
        Learning rate multiplier (actual lr = base_lr × this value).
    """
    # Clamp to 1 so step 0 never causes division by zero
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


def build_noam_scheduler(
    optimizer:     Optimizer,
    d_model:       int,
    warmup_steps:  int,
    last_step:     int = -1,
) -> LambdaLR:
    """
    Wrap `noam_lambda` in a PyTorch LambdaLR scheduler.

    Important: set the optimizer's initial lr to 1.0.  The Noam formula
    already encodes the absolute learning rate; using any other base lr
    scales the entire schedule by a constant factor, which is equivalent
    but confusing.  Recommended optimizer construction:

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1.0,                 # <-- base lr, Noam handles the rest
            betas=(0.9, 0.98),
            eps=1e-9,
        )

    Parameters
    ----------
    optimizer    : Optimizer   The Adam optimizer to wrap.
    d_model      : int         Model dimension from config.
    warmup_steps : int         Warmup steps from config (paper default: 4000).
    last_step    : int         Last completed step for resuming from checkpoint.
                               Pass the saved `global_step - 1` when resuming.

    Returns
    -------
    LambdaLR
        Scheduler.  Call `scheduler.step()` once per optimizer step
        (NOT once per epoch).
    """
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: noam_lambda(step, d_model, warmup_steps),
        last_epoch=last_step,
    )

    # Log the peak learning rate so it's easy to verify the schedule
    peak_step = warmup_steps
    peak_lr   = noam_lambda(peak_step, d_model, warmup_steps)
    logger.info(
        "Noam LR schedule: d_model=%d  warmup=%d  "
        "peak_lr=%.2e (at step %d)",
        d_model, warmup_steps, peak_lr, peak_step,
    )

    return scheduler
