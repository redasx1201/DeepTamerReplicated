"""
importance_weights.py
Implements Equation (4) from the Deep TAMER paper:

    w(ts, te, tf) = integral from (tf - te) to (tf - ts) of f_delay(t) dt

where f_delay is the assumed distribution of human reaction delay.

We use f_delay = Uniform[delay_min, delay_max] (default [0.2, 4.0] seconds),
matching the paper's choice.

Intuition:
- ts, te: start/end time of a state-action pair (s, a)
- tf:     time at which the human's feedback signal arrived
- The delay d = tf - t_action must lie in [delay_min, delay_max] for the
  feedback to plausibly apply to that action.
- For a state-action interval [ts, te], the delay d lies in [tf-te, tf-ts].
- w = fraction of [tf-te, tf-ts] that overlaps with [delay_min, delay_max].
"""


def compute_importance_weight(
    ts: float,
    te: float,
    tf: float,
    delay_min: float = 0.2,
    delay_max: float = 4.0,
) -> float:
    """
    Compute importance weight w(ts, te, tf).

    Args:
        ts: start time of the state-action pair
        te: end time of the state-action pair
        tf: time the human feedback was observed
        delay_min: minimum plausible human reaction delay (seconds)
        delay_max: maximum plausible human reaction delay (seconds)

    Returns:
        Scalar weight in [0, 1]. Zero means the feedback cannot plausibly
        apply to this state-action pair.
    """
    # If feedback arrived before the state-action pair started, weight = 0
    if tf < ts:
        return 0.0

    # Delay range implied by the state-action interval [ts, te]
    delay_lo = tf - te   # minimum possible delay (feedback earliest)
    delay_hi = tf - ts   # maximum possible delay (feedback latest)

    # Overlap of [delay_lo, delay_hi] with [delay_min, delay_max]
    overlap_lo = max(delay_lo, delay_min)
    overlap_hi = min(delay_hi, delay_max)

    if overlap_hi <= overlap_lo:
        return 0.0

    # Normalise by the total support of the uniform distribution
    return (overlap_hi - overlap_lo) / (delay_max - delay_min)
