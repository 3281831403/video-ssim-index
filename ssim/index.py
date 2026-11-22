"""SSIM and multi-scale SSIM over plain Python images.

An image is a non-empty, rectangular ``list[list[float]]``.  All values are
finite.  A local window is the clipped rectangle ``(2*radius + 1)`` square
around a pixel, so pixels near a border contain fewer valid samples.

Normalization contract
----------------------
The stabilizers are ``C1 = (K1 * L) ** 2`` and ``C2 = (K2 * L) ** 2`` with
``K1 = 0.01`` and ``K2 = 0.03``.  ``L`` is inferred once per image pair from
the supplied pixel values: ``L = 1.0`` when every value lies in ``[0.0, 1.0]``,
otherwise ``L = 255.0``.  Thus the same picture passed as 8-bit samples or as
values divided by 255 must produce the same index.

The public ``ssim_map`` arguments are the normalized coefficients
``c1 = 0.01 ** 2`` and ``c2 = 0.03 ** 2`` (the values for ``L = 1``);
``ssim_map`` multiplies each by ``L ** 2`` before evaluating the formula.
Comparisons in the tests use an absolute tolerance of ``1e-9``.

All local moments are population moments: every variance and covariance
divides by the number of valid pixels in that window.  Variances are clamped
at zero to absorb floating-point residue; covariance keeps its sign.
"""

import math

# Normalized stabilizer coefficients (the values for an input range of 1.0).
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2

# Input range used for 8-bit (or otherwise non-normalized) samples.
_FULL_RANGE = 255.0


def _is_finite_number(value):
    """Return whether ``value`` is a real scalar that is finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError):
        return False


def _check_image(img, name="image"):
    """Validate an image and return ``(height, width)``.

    Raises ``ValueError`` if ``img`` is not a non-empty, rectangular sequence
    of rows of finite numbers.
    """
    if not isinstance(img, (list, tuple)) or len(img) == 0:
        raise ValueError(f"{name} must be a non-empty list of rows")

    width = None
    for row_index, row in enumerate(img):
        if not isinstance(row, (list, tuple)) or len(row) == 0:
            raise ValueError(
                f"{name}[{row_index}] must be a non-empty list of values"
            )
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} must be rectangular")
        for column_index, value in enumerate(row):
            if not _is_finite_number(value):
                raise ValueError(
                    f"{name}[{row_index}][{column_index}] "
                    "must be a finite number"
                )
    return len(img), width


def _check_radius(radius):
    """Validate a window radius, raising ``ValueError`` when invalid."""
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    return radius


def _check_stabilizer(value, name):
    """Validate a normalized stabilizer coefficient."""
    if not _is_finite_number(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _window_bounds(index, size, radius):
    """Return the inclusive clipped window bounds around ``index``."""
    return max(0, index - radius), min(size - 1, index + radius)


def _infer_range(a, b):
    """Infer ``L`` from the pixel values of the pair.

    ``L = 1.0`` when every value of both images lies in ``[0.0, 1.0]``,
    otherwise ``L = 255.0``.
    """
    for img in (a, b):
        for row in img:
            for value in row:
                if value < 0.0 or value > 1.0:
                    return _FULL_RANGE
    return 1.0


def _clamp_score(value):
    """Clip a score to ``[-1, 1]`` to absorb floating-point overshoot."""
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


def box_filter(img, radius):
    """Return the edge-normalized mean filter of ``img``.

    Parameters
    ----------
    img:
        Non-empty rectangular ``list[list[float]]``.
    radius:
        Non-negative integer.  The window is ``2*radius + 1`` square.

    Returns
    -------
    list[list[float]]
        An image with the same shape as ``img``.  At every output pixel,
        average only the samples of the clipped window that actually exist
        and divide by that valid sample count, never by the full kernel area.

    Raises
    ------
    ValueError
        If ``img`` is not a non-empty rectangular image or ``radius`` is not
        a non-negative integer.
    """
    height, width = _check_image(img, "img")
    _check_radius(radius)

    result = [[0.0] * width for _ in range(height)]
    for row_index in range(height):
        first_row, last_row = _window_bounds(row_index, height, radius)
        for column_index in range(width):
            first_column, last_column = _window_bounds(
                column_index, width, radius
            )
            total = 0.0
            for inner_row in range(first_row, last_row + 1):
                source_row = img[inner_row]
                for inner_column in range(first_column, last_column + 1):
                    total += source_row[inner_column]
            count = (last_row - first_row + 1) * (
                last_column - first_column + 1
            )
            result[row_index][column_index] = total / count
    return result


def local_stats(a, b, radius):
    """Return local population statistics for two same-shaped images.

    Every window is clipped at the image border and normalized by its number
    of valid samples ``N``.  The returned dictionary has exactly these keys,
    each mapping to a same-shaped ``list[list[float]]``:

    ``"mu_a"``
        Local mean of ``a``.
    ``"mu_b"``
        Local mean of ``b``.
    ``"var_a"``
        ``E[a*a] - mu_a*mu_a`` over the valid samples, clamped at zero.
    ``"var_b"``
        ``E[b*b] - mu_b*mu_b`` over the valid samples, clamped at zero.
    ``"cov"``
        ``E[a*b] - mu_a*mu_b`` over the same ``N`` samples.  It is not
        clamped and its sign is preserved.

    ``a`` and ``b`` must have the same shape, and ``radius`` must be a
    non-negative integer.  Invalid input raises ``ValueError``.
    """
    height_a, width_a = _check_image(a, "a")
    height_b, width_b = _check_image(b, "b")
    if height_a != height_b or width_a != width_b:
        raise ValueError("a and b must have the same shape")
    _check_radius(radius)

    mu_a = [[0.0] * width_a for _ in range(height_a)]
    mu_b = [[0.0] * width_a for _ in range(height_a)]
    var_a = [[0.0] * width_a for _ in range(height_a)]
    var_b = [[0.0] * width_a for _ in range(height_a)]
    cov = [[0.0] * width_a for _ in range(height_a)]

    for row_index in range(height_a):
        first_row, last_row = _window_bounds(row_index, height_a, radius)
        for column_index in range(width_a):
            first_column, last_column = _window_bounds(
                column_index, width_a, radius
            )
            count = (last_row - first_row + 1) * (
                last_column - first_column + 1
            )

            sum_a = 0.0
            sum_b = 0.0
            for inner_row in range(first_row, last_row + 1):
                row_a = a[inner_row]
                row_b = b[inner_row]
                for inner_column in range(first_column, last_column + 1):
                    sum_a += row_a[inner_column]
                    sum_b += row_b[inner_column]

            mean_a = sum_a / count
            mean_b = sum_b / count

            # Centered (two-pass) population moments: mathematically identical
            # to E[x*x] - mu*mu but free of catastrophic cancellation, so a
            # constant window yields exactly zero variance and covariance.
            sum_aa = 0.0
            sum_bb = 0.0
            sum_ab = 0.0
            for inner_row in range(first_row, last_row + 1):
                row_a = a[inner_row]
                row_b = b[inner_row]
                for inner_column in range(first_column, last_column + 1):
                    delta_a = row_a[inner_column] - mean_a
                    delta_b = row_b[inner_column] - mean_b
                    sum_aa += delta_a * delta_a
                    sum_bb += delta_b * delta_b
                    sum_ab += delta_a * delta_b

            variance_a = sum_aa / count
            variance_b = sum_bb / count
            covariance = sum_ab / count

            mu_a[row_index][column_index] = mean_a
            mu_b[row_index][column_index] = mean_b
            var_a[row_index][column_index] = (
                variance_a if variance_a > 0.0 else 0.0
            )
            var_b[row_index][column_index] = (
                variance_b if variance_b > 0.0 else 0.0
            )
            cov[row_index][column_index] = covariance

    return {
        "mu_a": mu_a,
        "mu_b": mu_b,
        "var_a": var_a,
        "var_b": var_b,
        "cov": cov,
    }


def ssim_map(a, b, radius, c1, c2):
    """Return the per-pixel SSIM map for two same-shaped images.

    At each pixel the local means, variances and covariance come from
    :func:`local_stats`.  With ``C1 = c1 * L**2`` and ``C2 = c2 * L**2``::

        ((2*mu_a*mu_b + C1) * (2*cov + C2)) /
        ((mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2))

    ``c1`` and ``c2`` are the normalized coefficients for ``L = 1`` (the
    caller should normally pass ``0.01**2`` and ``0.03**2``), not raw 8-bit
    scale constants.  ``L`` is inferred as described in the module docstring:
    ``1.0`` for an image whose values all lie in ``[0.0, 1.0]``, otherwise
    ``255.0``.  The result has the same shape as ``a``.
    """
    _check_stabilizer(c1, "c1")
    _check_stabilizer(c2, "c2")

    stats = local_stats(a, b, radius)

    dynamic_range = _infer_range(a, b)
    scale = dynamic_range * dynamic_range
    stabilizer_1 = c1 * scale
    stabilizer_2 = c2 * scale

    result = []
    for row_index in range(len(a)):
        output_row = []
        for column_index in range(len(a[row_index])):
            mean_a = stats["mu_a"][row_index][column_index]
            mean_b = stats["mu_b"][row_index][column_index]
            variance_a = stats["var_a"][row_index][column_index]
            variance_b = stats["var_b"][row_index][column_index]
            covariance = stats["cov"][row_index][column_index]

            numerator = (2.0 * mean_a * mean_b + stabilizer_1) * (
                2.0 * covariance + stabilizer_2
            )
            denominator = (
                mean_a * mean_a + mean_b * mean_b + stabilizer_1
            ) * (variance_a + variance_b + stabilizer_2)

            if denominator == 0.0:
                value = 1.0 if numerator == 0.0 else 0.0
            else:
                value = numerator / denominator
            output_row.append(value)
        result.append(output_row)
    return result


def _weighted_mean_ssim(a, b, radius, c1, c2):
    """Edge-area-weighted mean of the SSIM map at ``radius``."""
    values = ssim_map(a, b, radius, c1, c2)
    height = len(values)
    width = len(values[0])

    numerator = 0.0
    denominator = 0.0
    for row_index in range(height):
        first_row, last_row = _window_bounds(row_index, height, radius)
        for column_index in range(width):
            first_column, last_column = _window_bounds(
                column_index, width, radius
            )
            weight = (last_row - first_row + 1) * (
                last_column - first_column + 1
            )
            numerator += values[row_index][column_index] * weight
            denominator += weight

    return _clamp_score(numerator / denominator)


def mean_ssim(a, b):
    """Return the border-weighted mean of the SSIM map, in ``[-1, 1]``.

    The local window radius is fixed at ``1`` and the normalized stabilizers
    are ``c1 = 0.01**2`` and ``c2 = 0.03**2``.  The per-pixel map is averaged
    with weights equal to the number of valid samples in that pixel's clipped
    window, so a corner pixel (4 valid samples at radius 1) does not carry the
    same weight as an interior pixel (9 valid samples).  The final value is
    clipped to ``[-1, 1]`` only to absorb floating-point overshoot.  Invalid
    input raises ``ValueError``.
    """
    return _weighted_mean_ssim(a, b, 1, _C1, _C2)


def ms_ssim(a, b, weights):
    """Return multi-scale SSIM, in ``[-1, 1]``.

    ``weights`` is a non-empty sequence of non-negative level weights.  Level
    zero is the original image pair.  Each later level is produced by applying
    :func:`downsample` to both images from the previous level.  The score at a
    level is :func:`mean_ssim` on that level's pair.  Weights are normalized
    internally by their sum before combining the level scores; consequently
    passing ``[1, 1]`` and ``[0.5, 0.5]`` must give the same answer, and extra
    zero-weight levels must not change the result.  A zero or non-finite weight
    sum raises ``ValueError``.
    """
    if isinstance(weights, (str, bytes)):
        raise ValueError("weights must be a non-empty sequence of numbers")
    try:
        weight_list = list(weights)
    except TypeError:
        raise ValueError("weights must be a non-empty sequence of numbers")

    if len(weight_list) == 0:
        raise ValueError("weights must be a non-empty sequence of numbers")

    cleaned = []
    for weight in weight_list:
        if not _is_finite_number(weight) or weight < 0:
            raise ValueError(
                "weights must be finite non-negative numbers"
            )
        cleaned.append(float(weight))

    total_weight = math.fsum(cleaned)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("weights must have a positive finite sum")

    level_a = a
    level_b = b
    contributions = []
    for level, weight in enumerate(cleaned):
        if level > 0:
            level_a = downsample(level_a)
            level_b = downsample(level_b)
        score = _weighted_mean_ssim(level_a, level_b, 1, _C1, _C2)
        contributions.append(weight * score)

    return _clamp_score(math.fsum(contributions) / total_weight)


def downsample(img):
    """Return the 2x2 mean downsample of ``img``.

    Output pixel ``(i, j)`` is the mean of input rows ``2*i`` and ``2*i + 1``
    and columns ``2*j`` and ``2*j + 1``.  The output height is
    ``ceil(height / 2)`` and the output width is ``ceil(width / 2)``.  At an
    odd edge only the existing samples enter the mean, so the final row and/or
    column is preserved rather than dropped.  The block average is used; this
    is not point sampling.
    """
    height, width = _check_image(img, "img")

    output_height = (height + 1) // 2
    output_width = (width + 1) // 2
    result = [[0.0] * output_width for _ in range(output_height)]

    for output_row in range(output_height):
        first_row = 2 * output_row
        last_row = min(first_row + 1, height - 1)
        row_count = last_row - first_row + 1
        for output_column in range(output_width):
            first_column = 2 * output_column
            last_column = min(first_column + 1, width - 1)
            column_count = last_column - first_column + 1

            total = 0.0
            for inner_row in range(first_row, last_row + 1):
                source_row = img[inner_row]
                for inner_column in range(first_column, last_column + 1):
                    total += source_row[inner_column]
            result[output_row][output_column] = total / (
                row_count * column_count
            )
    return result


def _scan_image(img, name):
    """Collect validation errors for ``img``.

    Returns ``(height, width, errors)`` where ``height``/``width`` are ``None``
    when they could not be determined.  Never raises.
    """
    errors = []
    if not isinstance(img, (list, tuple)):
        return None, None, [f"{name} must be a sequence of rows"]
    if len(img) == 0:
        return 0, 0, [f"{name} must be non-empty"]

    width = None
    for row_index, row in enumerate(img):
        if not isinstance(row, (list, tuple)):
            errors.append(f"{name}[{row_index}] is not a sequence")
            continue
        if len(row) == 0:
            errors.append(f"{name}[{row_index}] is empty")
            continue
        if width is None:
            width = len(row)
        elif len(row) != width:
            errors.append(
                f"{name}[{row_index}] has length {len(row)}, "
                f"expected {width}"
            )
        for column_index, value in enumerate(row):
            if not _is_finite_number(value):
                errors.append(
                    f"{name}[{row_index}][{column_index}] "
                    "is not a finite number"
                )
    return len(img), width, errors


def validate(a, b, radius):
    """Return a list of human-readable validation errors.

    An empty list means the pair is valid.  The checks cover:

    * ``a`` and ``b`` are non-empty rectangular images of finite floats;
    * ``a`` and ``b`` have identical height and width;
    * ``radius`` is a non-negative integer smaller than both image dimensions;
    * neither image contains ``NaN`` (or any other non-finite value).

    The function never raises for malformed values; it reports them in the
    returned list.  Messages are not part of the numerical contract, but a
    failure must produce at least one string.
    """
    errors = []

    height_a, width_a, errors_a = _scan_image(a, "a")
    height_b, width_b, errors_b = _scan_image(b, "b")
    errors.extend(errors_a)
    errors.extend(errors_b)

    shapes_known = (
        height_a is not None
        and width_a is not None
        and height_b is not None
        and width_b is not None
    )
    if shapes_known and (
        height_a != height_b or width_a != width_b
    ):
        errors.append("a and b must have identical dimensions")

    if isinstance(radius, bool) or not isinstance(radius, int):
        errors.append("radius must be a non-negative integer")
    elif radius < 0:
        errors.append("radius must be a non-negative integer")
    elif shapes_known:
        if (
            radius >= height_a
            or radius >= width_a
            or radius >= height_b
            or radius >= width_b
        ):
            errors.append(
                "radius must be smaller than both image dimensions"
            )

    return errors


def ssim_batch(pairs, config):
    """Score a sequence of image pairs independently.

    ``pairs`` is an iterable of ``(a, b)`` pairs.  ``config`` is a mapping;
    ``config.get("radius", 1)`` selects the local-window radius for every
    pair.  The return value is::

        {
            "scores": [float | None, ...],
            "errors": [list[str], ...],
        }

    ``scores[k]`` is the edge-area-weighted mean SSIM of pair ``k`` at the
    configured radius (using ``c1 = 0.01**2`` and ``c2 = 0.03**2``), clipped
    to ``[-1, 1]``.  If pair ``k`` is invalid, ``scores[k]`` is ``None`` and
    ``errors[k]`` contains the messages from :func:`validate`.  Every pair is
    evaluated from its own statistics: no mean, variance, covariance or
    scratch buffer may be carried over from a previous pair.
    """
    radius = 1
    if config is not None and hasattr(config, "get"):
        radius = config.get("radius", 1)

    scores = []
    errors = []
    for pair in pairs:
        a, b = pair
        messages = validate(a, b, radius)
        if messages:
            scores.append(None)
            errors.append(messages)
            continue

        try:
            # Each pair is scored from freshly computed statistics; nothing is
            # shared with any previous pair.
            score = _weighted_mean_ssim(a, b, radius, _C1, _C2)
        except ValueError as error:
            scores.append(None)
            errors.append([str(error)])
        else:
            scores.append(_clamp_score(score))
            errors.append([])

    return {"scores": scores, "errors": errors}
