import math

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


# ============================================================
# CONFIGURATION
# ============================================================

RESONATOR_NAME = "resonator"
WIDTH_OPTION = "trace_width"


# ============================================================
# UNIT CONVERSION
# ============================================================

def to_meters(value):
    """
    Convert a dimensional value to meters.

    Examples
    --------
    10e-6
    "10um"
    "10 µm"
    "0.01mm"
    "10000nm"
    """

    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = (
        str(value)
        .strip()
        .lower()
        .replace("µ", "u")
        .replace(" ", "")
    )

    if text.endswith("um"):
        return float(text[:-2]) * 1e-6

    if text.endswith("mm"):
        return float(text[:-2]) * 1e-3

    if text.endswith("nm"):
        return float(text[:-2]) * 1e-9

    if text.endswith("m"):
        return float(text[:-1])

    return float(text)


# ============================================================
# UPDATE RESONATOR WIDTH
# ============================================================

def update_resonator_width(
    design,
    width,
    component_name=RESONATOR_NAME,
    width_option=WIDTH_OPTION,
):
    """
    Change the resonator trace width and rebuild the geometry.
    """

    width_m = to_meters(width)
    width_um = width_m * 1e6

    width_string = f"{width_um:.9f}um"

    if not hasattr(design, "components"):
        raise AttributeError(
            "The supplied design has no 'components' attribute."
        )

    if component_name not in design.components:
        raise KeyError(
            f"Component '{component_name}' not found.\n"
            f"Available components: "
            f"{list(design.components.keys())}"
        )

    component = design.components[component_name]

    if width_option not in component.options:
        raise KeyError(
            f"Option '{width_option}' not found in "
            f"component '{component_name}'.\n"
            f"Available options: "
            f"{list(component.options.keys())}"
        )

    old_width = component.options[width_option]

    print("\nUpdating resonator geometry")
    print(f"    Component : {component_name}")
    print(f"    Parameter : {width_option}")
    print(f"    Old width : {old_width}")
    print(f"    New width : {width_string}")

    component.options[width_option] = width_string

    # Rebuild the modified component.
    if hasattr(component, "rebuild"):
        component.rebuild()

    # Some design implementations expose a full rebuild.
    # Try it, but do not fail if unsupported.
    if hasattr(design, "rebuild"):
        try:
            design.rebuild()
        except Exception as exc:
            print(
                f"    Full design rebuild skipped/warning: {exc}"
            )

    actual_width = (
        design.components[component_name]
        .options[width_option]
    )

    print(f"    Verified  : {actual_width}")

    return width_m


# ============================================================
# EXTRACT PALACE FREQUENCY
# ============================================================

def extract_palace_frequency(
    simulation,
    mode_index=0,
):
    """
    Extract Re{f} from Palace.

    Supports:
        1. Named columns such as Re{f} (GHz)
        2. Unnamed columns [0, 1, 2, ...]

    For the unnamed Palace table observed here:
        column 0 -> mode number
        column 1 -> Re{f} (GHz)

    Returns
    -------
    float
        Frequency in Hz.
    """

    data = simulation.retrieve_data()

    if data is None:
        raise RuntimeError(
            "Palace retrieve_data() returned None."
        )

    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as exc:
            raise RuntimeError(
                "Could not convert Palace output "
                "to a pandas DataFrame."
            ) from exc

    if data.empty:
        raise RuntimeError(
            "Palace returned an empty eigenmode table."
        )

    print("\nPalace eigenmode data:")
    print(data)

    if mode_index < 0 or mode_index >= len(data):
        raise IndexError(
            f"mode_index={mode_index} is invalid. "
            f"Palace returned {len(data)} rows."
        )

    frequency_column = None

    # --------------------------------------------------------
    # Try to find a named Re{f} column.
    # --------------------------------------------------------

    for column in data.columns:

        normalized = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "")
        )

        if (
            "re{f}" in normalized
            or "re(f)" in normalized
        ):
            frequency_column = column
            break

    # --------------------------------------------------------
    # Fallback for unnamed Palace table.
    #
    # Based on the output:
    #     0 = mode m
    #     1 = Re{f} (GHz)
    # --------------------------------------------------------

    if frequency_column is None:

        if len(data.columns) < 2:
            raise RuntimeError(
                "Could not identify the Palace "
                "frequency column."
            )

        frequency_column = data.columns[1]

        print(
            f"\nNo named Re{{f}} column found. "
            f"Using column '{frequency_column}' "
            f"as Re{{f}} (GHz)."
        )

    raw_frequency = (
        data[frequency_column]
        .iloc[mode_index]
    )

    try:
        frequency_ghz = float(raw_frequency)

    except Exception as exc:

        raise RuntimeError(
            f"Could not convert Palace frequency "
            f"'{raw_frequency}' to float."
        ) from exc

    if not math.isfinite(frequency_ghz):

        raise RuntimeError(
            f"Invalid Palace frequency: "
            f"{frequency_ghz}"
        )

    if frequency_ghz <= 0:

        raise RuntimeError(
            f"Non-positive Palace frequency: "
            f"{frequency_ghz} GHz"
        )

    frequency_hz = frequency_ghz * 1e9

    print("\nSelected eigenmode")
    print(f"    Mode index : {mode_index}")
    print(
        f"    Frequency  : "
        f"{frequency_ghz:.9f} GHz"
    )

    return frequency_hz


# ============================================================
# RUN ONE PALACE SIMULATION
# ============================================================

def run_palace_at_width(
    width,
    design,
    simulation,
    mode_index=0,
):
    """
    Run one Palace simulation at a specified resonator width.

    IMPORTANT:
        No adaptive meshing.
        No fidelity switching.
        No solver-option modification.

    The existing simulation configuration is reused.

    Sequence:
        update trace_width
            ->
        rebuild geometry
            ->
        prepare_simulation()
            ->
        simulation.run()
            ->
        extract frequency
    """

    width_m = to_meters(width)
    width_um = width_m * 1e6

    # --------------------------------------------------------
    # 1. UPDATE GEOMETRY
    # --------------------------------------------------------

    update_resonator_width(
        design=design,
        width=width_m,
    )

    # --------------------------------------------------------
    # 2. KEEP SIMULATION ATTACHED TO CURRENT DESIGN
    # --------------------------------------------------------

    if hasattr(simulation, "metal_design"):

        try:
            simulation.metal_design = design
        except Exception:
            pass

    if hasattr(simulation, "design"):

        try:
            simulation.design = design
        except Exception:
            pass

    # --------------------------------------------------------
    # 3. PREPARE
    # --------------------------------------------------------

    print(
        f"\nPreparing Palace simulation "
        f"at {width_um:.9f} µm..."
    )

    simulation.prepare_simulation()

    sim_config = getattr(
        simulation,
        "_sim_config",
        "",
    )

    if sim_config == "":

        raise RuntimeError(
            "prepare_simulation() completed, "
            "but Palace _sim_config is empty."
        )

    print("Palace simulation prepared.")

    # --------------------------------------------------------
    # 4. RUN
    # --------------------------------------------------------

    print("\nRunning Palace...")

    simulation.run()

    # --------------------------------------------------------
    # 5. EXTRACT FREQUENCY
    # --------------------------------------------------------

    frequency = extract_palace_frequency(
        simulation=simulation,
        mode_index=mode_index,
    )

    print("\nPalace evaluation complete")
    print(
        f"    Width     : "
        f"{width_um:.9f} µm"
    )
    print(
        f"    Frequency : "
        f"{frequency / 1e9:.9f} GHz"
    )

    return frequency


# ============================================================
# BRENT MINIMIZATION
# ============================================================

def brent_minimize(
    function,
    lower,
    upper,
    tolerance=1e-9,
    max_iterations=100,
):
    """
    Bounded derivative-free Brent minimization.

    Combines:
        - Successive parabolic interpolation
        - Golden-section fallback

    Palace is NOT called inside this function.
    """

    a = float(lower)
    b = float(upper)

    if a >= b:
        raise ValueError(
            "Lower bound must be smaller than upper bound."
        )

    golden = (
        3.0 - math.sqrt(5.0)
    ) / 2.0

    tiny = 1e-15

    x = a + golden * (b - a)

    w = x
    v = x

    fx = float(function(x))
    fw = fx
    fv = fx

    d = 0.0
    e = 0.0

    converged = False

    for iteration in range(
        1,
        max_iterations + 1,
    ):

        midpoint = 0.5 * (a + b)

        tol1 = (
            tolerance
            + tiny * abs(x)
        )

        tol2 = 2.0 * tol1

        # ----------------------------------------------------
        # CONVERGENCE
        # ----------------------------------------------------

        if abs(x - midpoint) <= (
            tol2
            - 0.5 * (b - a)
        ):

            converged = True
            break

        previous_e = e

        parabolic_accepted = False

        # ----------------------------------------------------
        # SUCCESSIVE PARABOLIC INTERPOLATION
        # ----------------------------------------------------

        if abs(e) > tol1:

            r = (
                (x - w)
                * (fx - fv)
            )

            q = (
                (x - v)
                * (fx - fw)
            )

            p = (
                (x - v) * q
                - (x - w) * r
            )

            q = 2.0 * (q - r)

            if q > 0:
                p = -p
            else:
                q = -q

            e = d

            if (
                q > 0
                and abs(p)
                < abs(
                    0.5
                    * q
                    * previous_e
                )
                and p > q * (a - x)
                and p < q * (b - x)
            ):

                d = p / q

                candidate = x + d

                if (
                    candidate - a < tol2
                    or
                    b - candidate < tol2
                ):

                    direction = midpoint - x

                    if direction == 0:
                        direction = 1.0

                    d = math.copysign(
                        tol1,
                        direction,
                    )

                parabolic_accepted = True

        # ----------------------------------------------------
        # GOLDEN-SECTION FALLBACK
        # ----------------------------------------------------

        if not parabolic_accepted:

            if x < midpoint:
                e = b - x
            else:
                e = a - x

            d = golden * e

        # ----------------------------------------------------
        # MINIMUM STEP
        # ----------------------------------------------------

        if abs(d) >= tol1:

            u = x + d

        else:

            direction = (
                d
                if d != 0
                else midpoint - x
            )

            if direction == 0:
                direction = 1.0

            u = (
                x
                + math.copysign(
                    tol1,
                    direction,
                )
            )

        u = min(
            max(u, a),
            b,
        )

        fu = float(
            function(u)
        )

        # ----------------------------------------------------
        # UPDATE BRENT STATE
        # ----------------------------------------------------

        if fu <= fx:

            if u >= x:
                a = x
            else:
                b = x

            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu

        else:

            if u < x:
                a = u
            else:
                b = u

            if (
                fu <= fw
                or w == x
            ):

                v, fv = w, fw
                w, fw = u, fu

            elif (
                fu <= fv
                or v == x
                or v == w
            ):

                v, fv = u, fu

    else:
        iteration = max_iterations

    return (
        x,
        fx,
        iteration,
        converged,
    )


# ============================================================
# BUILD PCHIP SURROGATE
# ============================================================

def build_surrogate(
    widths,
    frequencies,
):
    """
    Build a shape-preserving PCHIP surrogate.

    PCHIP is used instead of a global quadratic because it
    avoids artificial polynomial overshoot and nonsense
    extrapolation.

    Width:
        meters -> internally converted to µm

    Frequency:
        Hz -> internally converted to GHz
    """

    if PchipInterpolator is None:

        raise ImportError(
            "SciPy is required for PCHIP.\n"
            "Install it using:\n"
            "pip install scipy"
        )

    x = (
        np.asarray(
            widths,
            dtype=float,
        )
        * 1e6
    )

    y = (
        np.asarray(
            frequencies,
            dtype=float,
        )
        / 1e9
    )

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    # --------------------------------------------------------
    # REMOVE DUPLICATE WIDTHS
    # --------------------------------------------------------

    unique_x = []
    unique_y = []

    for xi, yi in zip(x, y):

        if (
            len(unique_x) == 0
            or abs(
                xi - unique_x[-1]
            ) > 1e-12
        ):

            unique_x.append(
                float(xi)
            )

            unique_y.append(
                float(yi)
            )

        else:

            # Keep latest result at duplicate width.
            unique_y[-1] = float(yi)

    if len(unique_x) < 2:

        raise RuntimeError(
            "At least two unique Palace samples "
            "are required to build the surrogate."
        )

    interpolator = PchipInterpolator(
        unique_x,
        unique_y,
        extrapolate=False,
    )

    minimum_width = (
        min(unique_x)
        * 1e-6
    )

    maximum_width = (
        max(unique_x)
        * 1e-6
    )

    def surrogate_frequency(width):

        width = float(width)

        if (
            width < minimum_width
            or width > maximum_width
        ):

            return float("nan")

        width_um = (
            width * 1e6
        )

        value = interpolator(
            width_um
        )

        value = float(value)

        if not math.isfinite(value):

            return float("nan")

        return value * 1e9

    return (
        surrogate_frequency,
        minimum_width,
        maximum_width,
    )


# ============================================================
# MAIN OPTIMIZER
# ============================================================

def optimize_resonator_width(
    target_frequency,
    initial_width,
    design,
    simulation,
    lower_bound,
    upper_bound,
    mode_index=0,
    frequency_tolerance=1e6,
    width_tolerance=1e-9,
    max_palace_calls=6,
    max_brent_iterations=100,
    minimum_frequency_variation=1e3,
):
    """
    Fixed-mesh surrogate-assisted Brent optimizer.

    NO adaptive meshing.
    NO fidelity switching.
    NO repeated final validation.

    Workflow
    --------

        3 Palace samples
              |
              v
        PCHIP surrogate
              |
              v
        Brent minimization
        (cheap, no Palace)
              |
              v
        Palace validation
              |
              v
        Add real sample
              |
              v
        Rebuild PCHIP
              |
              v
        Repeat until:
            - frequency tolerance reached, or
            - Palace budget exhausted

    The returned final point is always an ACTUAL Palace sample.
    """

    target_frequency = float(
        target_frequency
    )

    initial_width = to_meters(
        initial_width
    )

    lower_bound = to_meters(
        lower_bound
    )

    upper_bound = to_meters(
        upper_bound
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if lower_bound >= upper_bound:

        raise ValueError(
            "lower_bound must be smaller "
            "than upper_bound."
        )

    if not (
        lower_bound
        <= initial_width
        <= upper_bound
    ):

        raise ValueError(
            "initial_width must lie inside "
            "the search bounds."
        )

    if max_palace_calls < 4:

        raise ValueError(
            "max_palace_calls must be at least 4."
        )

    # ========================================================
    # STORAGE
    # ========================================================

    sampled_widths = []
    sampled_frequencies = []

    prediction_history = []

    palace_calls = 0

    # ========================================================
    # PALACE EVALUATOR WITH CACHE
    # ========================================================

    def evaluate(width):

        nonlocal palace_calls

        width = float(width)

        width = min(
            max(
                width,
                lower_bound,
            ),
            upper_bound,
        )

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        for (
            old_width,
            old_frequency,
        ) in zip(
            sampled_widths,
            sampled_frequencies,
        ):

            if math.isclose(
                width,
                old_width,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):

                print(
                    "\nUsing cached Palace result:"
                )

                print(
                    f"    Width     : "
                    f"{old_width*1e6:.9f} µm"
                )

                print(
                    f"    Frequency : "
                    f"{old_frequency/1e9:.9f} GHz"
                )

                return old_frequency

        # ----------------------------------------------------
        # BUDGET
        # ----------------------------------------------------

        if palace_calls >= max_palace_calls:

            raise RuntimeError(
                "Maximum Palace call budget reached."
            )

        print(
            "\n"
            + "=" * 72
        )

        print(
            f"PALACE CALL "
            f"{palace_calls + 1}"
            f"/{max_palace_calls}"
        )

        print(
            f"WIDTH: "
            f"{width*1e6:.9f} µm"
        )

        print(
            "=" * 72
        )

        # ----------------------------------------------------
        # RUN
        # ----------------------------------------------------

        frequency = run_palace_at_width(
            width=width,
            design=design,
            simulation=simulation,
            mode_index=mode_index,
        )

        sampled_widths.append(
            width
        )

        sampled_frequencies.append(
            frequency
        )

        palace_calls += 1

        error = (
            target_frequency
            - frequency
        )

        print("\nPALACE RESULT")

        print(
            f"    Width     : "
            f"{width*1e6:.9f} µm"
        )

        print(
            f"    Frequency : "
            f"{frequency/1e9:.9f} GHz"
        )

        print(
            f"    Error     : "
            f"{error/1e6:.6f} MHz"
        )

        return frequency

    # ========================================================
    # HEADER
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "FIXED-MESH PCHIP + BRENT OPTIMIZATION"
    )

    print(
        "=" * 72
    )

    print(
        f"Target frequency : "
        f"{target_frequency/1e9:.9f} GHz"
    )

    print(
        f"Initial width    : "
        f"{initial_width*1e6:.9f} µm"
    )

    print(
        f"Search bounds    : "
        f"{lower_bound*1e6:.9f}"
        f" – "
        f"{upper_bound*1e6:.9f} µm"
    )

    print(
        f"Palace call cap  : "
        f"{max_palace_calls}"
    )

    print(
        "Adaptive meshing : DISABLED"
    )

    # ========================================================
    # INITIAL 3 PALACE SAMPLES
    #
    # For 5-20 µm and initial 10 µm:
    #
    #     7 µm
    #     10 µm
    #     13 µm
    # ========================================================

    span = (
        upper_bound
        - lower_bound
    )

    initial_points = [
        max(
            lower_bound,
            initial_width
            - 0.20 * span,
        ),

        initial_width,

        min(
            upper_bound,
            initial_width
            + 0.20 * span,
        ),
    ]

    # Remove duplicates.
    unique_initial_points = []

    for point in initial_points:

        if not any(
            math.isclose(
                point,
                existing,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for existing
            in unique_initial_points
        ):

            unique_initial_points.append(
                point
            )

    # If initial width is too close to a boundary and gives
    # fewer than 3 points, add midpoint/boundaries as needed.
    fallback_points = [
        lower_bound,
        0.5 * (
            lower_bound
            + upper_bound
        ),
        upper_bound,
    ]

    for point in fallback_points:

        if len(
            unique_initial_points
        ) >= 3:
            break

        if not any(
            math.isclose(
                point,
                existing,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for existing
            in unique_initial_points
        ):

            unique_initial_points.append(
                point
            )

    for width in unique_initial_points[:3]:

        evaluate(
            width
        )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    frequency_span = (
        max(sampled_frequencies)
        - min(sampled_frequencies)
    )

    print(
        "\n"
        + "-" * 72
    )

    print(
        "INITIAL SENSITIVITY CHECK"
    )

    print(
        "-" * 72
    )

    print(
        f"Frequency variation : "
        f"{frequency_span/1e6:.9f} MHz"
    )

    if (
        frequency_span
        < minimum_frequency_variation
    ):

        raise RuntimeError(
            "\nOptimization stopped.\n\n"
            "Changing trace_width produced almost "
            "no frequency variation.\n"
            "The optimizer will not fit Palace "
            "numerical noise.\n\n"
            f"Observed frequency span: "
            f"{frequency_span:.3f} Hz\n\n"
            "Check that changing trace_width actually "
            "changes the geometry used by Palace."
        )

    # ========================================================
    # HELPER:
    # BEST ACTUAL PALACE SAMPLE
    # ========================================================

    def get_best_actual():

        best_index = min(
            range(
                len(sampled_frequencies)
            ),
            key=lambda i: abs(
                target_frequency
                - sampled_frequencies[i]
            ),
        )

        return (
            sampled_widths[
                best_index
            ],
            sampled_frequencies[
                best_index
            ],
        )

    # ========================================================
    # MAIN OPTIMIZATION LOOP
    # ========================================================

    while (
        palace_calls
        < max_palace_calls
    ):

        # ----------------------------------------------------
        # BEST ACTUAL RESULT SO FAR
        # ----------------------------------------------------

        (
            best_width,
            best_frequency,
        ) = get_best_actual()

        best_error = (
            target_frequency
            - best_frequency
        )

        print(
            "\n"
            + "-" * 72
        )

        print(
            "CURRENT BEST ACTUAL PALACE RESULT"
        )

        print(
            "-" * 72
        )

        print(
            f"Width     : "
            f"{best_width*1e6:.9f} µm"
        )

        print(
            f"Frequency : "
            f"{best_frequency/1e9:.9f} GHz"
        )

        print(
            f"Error     : "
            f"{best_error/1e6:.6f} MHz"
        )

        # ----------------------------------------------------
        # CONVERGED?
        # ----------------------------------------------------

        if abs(
            best_error
        ) <= frequency_tolerance:

            print(
                "\nFrequency tolerance reached."
            )

            break

        # ----------------------------------------------------
        # SORT ACTUAL DATA
        # ----------------------------------------------------

        order = np.argsort(
            sampled_widths
        )

        widths = [
            sampled_widths[i]
            for i in order
        ]

        frequencies = [
            sampled_frequencies[i]
            for i in order
        ]

        # ----------------------------------------------------
        # BUILD PCHIP
        # ----------------------------------------------------

        (
            surrogate,
            sampled_lower,
            sampled_upper,
        ) = build_surrogate(
            widths,
            frequencies,
        )

        observed_min = min(
            frequencies
        )

        observed_max = max(
            frequencies
        )

        # ====================================================
        # CASE 1:
        # TARGET IS INSIDE OBSERVED FREQUENCY RANGE
        #
        # Use Brent on PCHIP.
        # ====================================================

        if (
            observed_min
            <= target_frequency
            <= observed_max
        ):

            print(
                "\nTarget lies inside the currently "
                "observed frequency range."
            )

            def objective(width):

                predicted = surrogate(
                    width
                )

                if not math.isfinite(
                    predicted
                ):

                    return float("inf")

                normalized_error = (
                    predicted
                    - target_frequency
                ) / 1e9

                return (
                    normalized_error
                    * normalized_error
                )

            (
                candidate,
                objective_value,
                brent_iterations,
                brent_converged,
            ) = brent_minimize(
                function=objective,
                lower=sampled_lower,
                upper=sampled_upper,
                tolerance=width_tolerance,
                max_iterations=max_brent_iterations,
            )

            predicted_frequency = surrogate(
                candidate
            )

            print(
                "\nBRENT PREDICTION"
            )

            print(
                f"    Width      : "
                f"{candidate*1e6:.9f} µm"
            )

            print(
                f"    Predicted  : "
                f"{predicted_frequency/1e9:.9f} GHz"
            )

            print(
                f"    Iterations : "
                f"{brent_iterations}"
            )

            print(
                f"    Converged  : "
                f"{brent_converged}"
            )

        # ====================================================
        # CASE 2:
        # TARGET ABOVE ALL OBSERVED FREQUENCIES
        #
        # Explore near the highest-frequency actual point.
        # ====================================================

        elif (
            target_frequency
            > observed_max
        ):

            print(
                "\nTarget is ABOVE the currently "
                "observed frequency range."
            )

            highest_index = int(
                np.argmax(
                    frequencies
                )
            )

            peak_width = widths[
                highest_index
            ]

            # ------------------------------------------------
            # Determine local trend around highest point.
            # ------------------------------------------------

            candidates = []

            if highest_index > 0:

                left_width = widths[
                    highest_index - 1
                ]

                left_midpoint = (
                    0.5
                    * (
                        left_width
                        + peak_width
                    )
                )

                candidates.append(
                    left_midpoint
                )

            if (
                highest_index
                < len(widths) - 1
            ):

                right_width = widths[
                    highest_index + 1
                ]

                right_midpoint = (
                    0.5
                    * (
                        peak_width
                        + right_width
                    )
                )

                candidates.append(
                    right_midpoint
                )

            # Explore unsampled outer region if peak is near
            # edge of current sampled domain.
            if (
                peak_width
                == max(widths)
                and
                peak_width
                < upper_bound
            ):

                candidates.append(
                    0.5
                    * (
                        peak_width
                        + upper_bound
                    )
                )

            if (
                peak_width
                == min(widths)
                and
                peak_width
                > lower_bound
            ):

                candidates.append(
                    0.5
                    * (
                        lower_bound
                        + peak_width
                    )
                )

            # Remove points too close to existing samples.
            candidates = [
                candidate
                for candidate
                in candidates
                if min(
                    abs(
                        candidate
                        - existing
                    )
                    for existing
                    in widths
                )
                > width_tolerance
            ]

            if candidates:

                # Choose the candidate closest to the
                # current best-frequency region.
                candidate = min(
                    candidates,
                    key=lambda value: abs(
                        value
                        - peak_width
                    ),
                )

            else:

                candidate = None

            predicted_frequency = float(
                "nan"
            )

            brent_iterations = 0
            brent_converged = False

        # ====================================================
        # CASE 3:
        # TARGET BELOW ALL OBSERVED FREQUENCIES
        # ====================================================

        else:

            print(
                "\nTarget is BELOW the currently "
                "observed frequency range."
            )

            lowest_index = int(
                np.argmin(
                    frequencies
                )
            )

            valley_width = widths[
                lowest_index
            ]

            candidates = []

            if lowest_index > 0:

                candidates.append(
                    0.5
                    * (
                        widths[
                            lowest_index - 1
                        ]
                        + valley_width
                    )
                )

            if (
                lowest_index
                < len(widths) - 1
            ):

                candidates.append(
                    0.5
                    * (
                        valley_width
                        + widths[
                            lowest_index + 1
                        ]
                    )
                )

            if (
                valley_width
                == max(widths)
                and
                valley_width
                < upper_bound
            ):

                candidates.append(
                    0.5
                    * (
                        valley_width
                        + upper_bound
                    )
                )

            if (
                valley_width
                == min(widths)
                and
                valley_width
                > lower_bound
            ):

                candidates.append(
                    0.5
                    * (
                        lower_bound
                        + valley_width
                    )
                )

            candidates = [
                candidate
                for candidate
                in candidates
                if min(
                    abs(
                        candidate
                        - existing
                    )
                    for existing
                    in widths
                )
                > width_tolerance
            ]

            if candidates:

                candidate = min(
                    candidates,
                    key=lambda value: abs(
                        value
                        - valley_width
                    ),
                )

            else:

                candidate = None

            predicted_frequency = float(
                "nan"
            )

            brent_iterations = 0
            brent_converged = False

        # ====================================================
        # IF CANDIDATE INVALID / DUPLICATE:
        # SAMPLE LARGEST UNSAMPLED GAP
        # ====================================================

        if candidate is not None:

            nearest_distance = min(
                abs(
                    candidate
                    - existing
                )
                for existing
                in widths
            )

        else:

            nearest_distance = 0.0

        if (
            candidate is None
            or nearest_distance
            <= width_tolerance
        ):

            print(
                "\nCandidate is duplicate or unavailable."
            )

            # Build intervals including global boundaries.
            all_points = sorted(
                set(
                    [lower_bound]
                    + widths
                    + [upper_bound]
                )
            )

            gaps = []

            for left, right in zip(
                all_points[:-1],
                all_points[1:],
            ):

                gap = (
                    right
                    - left
                )

                if (
                    gap
                    > 2.0
                    * width_tolerance
                ):

                    midpoint = (
                        0.5
                        * (
                            left
                            + right
                        )
                    )

                    if min(
                        abs(
                            midpoint
                            - existing
                        )
                        for existing
                        in widths
                    ) > width_tolerance:

                        gaps.append(
                            (
                                gap,
                                midpoint,
                            )
                        )

            if not gaps:

                print(
                    "\nNo useful unsampled region remains."
                )

                break

            # Explore largest unknown interval.
            (
                _,
                candidate,
            ) = max(
                gaps,
                key=lambda item: item[0],
            )

            predicted_frequency = float(
                "nan"
            )

        # ====================================================
        # VALIDATE CANDIDATE WITH ONE PALACE CALL
        # ====================================================

        print(
            "\nNEXT PALACE CANDIDATE"
        )

        print(
            f"    Width : "
            f"{candidate*1e6:.9f} µm"
        )

        if math.isfinite(
            predicted_frequency
        ):

            print(
                f"    PCHIP prediction : "
                f"{predicted_frequency/1e9:.9f} GHz"
            )

        actual_frequency = evaluate(
            candidate
        )

        actual_error = (
            target_frequency
            - actual_frequency
        )

        if math.isfinite(
            predicted_frequency
        ):

            surrogate_error = (
                actual_frequency
                - predicted_frequency
            )

        else:

            surrogate_error = float(
                "nan"
            )

        prediction_history.append(
            {
                "width":
                    candidate,

                "predicted_frequency":
                    predicted_frequency,

                "actual_frequency":
                    actual_frequency,

                "target_error":
                    actual_error,

                "surrogate_error":
                    surrogate_error,

                "brent_iterations":
                    brent_iterations,

                "brent_converged":
                    brent_converged,
            }
        )

        print(
            "\nVALIDATION"
        )

        print(
            f"    Actual frequency : "
            f"{actual_frequency/1e9:.9f} GHz"
        )

        print(
            f"    Target error     : "
            f"{actual_error/1e6:.6f} MHz"
        )

        if math.isfinite(
            surrogate_error
        ):

            print(
                f"    PCHIP error      : "
                f"{surrogate_error/1e6:.6f} MHz"
            )

        if abs(
            actual_error
        ) <= frequency_tolerance:

            print(
                "\nTarget frequency tolerance reached."
            )

            break

    # ========================================================
    # FINAL RESULT
    #
    # ALWAYS RETURN BEST ACTUAL PALACE SAMPLE.
    #
    # NO REDUNDANT FINAL PALACE RUN.
    # ========================================================

    (
        final_width,
        final_frequency,
    ) = get_best_actual()

    final_error = (
        target_frequency
        - final_frequency
    )

    converged = (
        abs(final_error)
        <= frequency_tolerance
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n"
        + "=" * 72
    )

    print(
        "OPTIMIZATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Final width     : "
        f"{final_width*1e6:.9f} µm"
    )

    print(
        f"Final frequency : "
        f"{final_frequency/1e9:.9f} GHz"
    )

    print(
        f"Final error     : "
        f"{final_error/1e6:.6f} MHz"
    )

    print(
        f"Palace calls    : "
        f"{palace_calls}"
    )

    print(
        f"Converged       : "
        f"{converged}"
    )

    if not converged:

        observed_min = min(
            sampled_frequencies
        )

        observed_max = max(
            sampled_frequencies
        )

        if (
            target_frequency
            > observed_max
        ):

            print(
                "\nWARNING:"
            )

            print(
                "The target frequency is above all "
                "frequencies observed by Palace."
            )

            print(
                "The requested target may not be "
                "reachable by changing trace_width "
                "within the selected bounds."
            )

        elif (
            target_frequency
            < observed_min
        ):

            print(
                "\nWARNING:"
            )

            print(
                "The target frequency is below all "
                "frequencies observed by Palace."
            )

    print(
        "=" * 72
    )

    return {
        "final_cpw_width":
            final_width,

        "final_frequency":
            final_frequency,

        "frequency_error":
            final_error,

        "converged":
            converged,

        "palace_calls":
            palace_calls,

        "sampled_widths":
            sampled_widths,

        "sampled_frequencies":
            sampled_frequencies,

        "prediction_history":
            prediction_history,

        "target_frequency":
            target_frequency,

        "mode_index":
            mode_index,

        "lower_bound":
            lower_bound,

        "upper_bound":
            upper_bound,

        "frequency_tolerance":
            frequency_tolerance,

        "width_tolerance":
            width_tolerance,

        "method":
            (
                "Fixed-Mesh Surrogate-Assisted Brent "
                "(PCHIP + SPI/GSS + Palace Validation)"
            ),
    }


# ============================================================
# PRINT PALACE SAMPLE TABLE
# ============================================================

def print_palace_samples(
    result,
):
    print(
        "\nPALACE SAMPLE HISTORY"
    )

    print(
        "-" * 82
    )

    print(
        "Call | Width (µm) | "
        "Frequency (GHz) | "
        "Target Error (MHz)"
    )

    print(
        "-" * 82
    )

    target = (
        result[
            "target_frequency"
        ]
    )

    for index, (
        width,
        frequency,
    ) in enumerate(
        zip(
            result[
                "sampled_widths"
            ],
            result[
                "sampled_frequencies"
            ],
        ),
        start=1,
    ):

        error = (
            target
            - frequency
        )

        print(
            f"{index:4d} | "
            f"{width*1e6:10.6f} | "
            f"{frequency/1e9:15.9f} | "
            f"{error/1e6:18.6f}"
        )


# ============================================================
# PLOT OPTIMIZATION
# ============================================================

def plot_optimization(
    result,
):
    import matplotlib.pyplot as plt

    widths = np.asarray(
        result[
            "sampled_widths"
        ],
        dtype=float,
    )

    frequencies = np.asarray(
        result[
            "sampled_frequencies"
        ],
        dtype=float,
    )

    order = np.argsort(
        widths
    )

    widths = widths[
        order
    ]

    frequencies = frequencies[
        order
    ]

    plt.figure(
        figsize=(9, 5)
    )

    # --------------------------------------------------------
    # ACTUAL PALACE POINTS
    # --------------------------------------------------------

    plt.scatter(
        widths * 1e6,
        frequencies / 1e9,
        s=55,
        label="Palace simulations",
    )

    # --------------------------------------------------------
    # PCHIP ONLY BETWEEN ACTUAL SAMPLED POINTS
    #
    # NO EXTRAPOLATION.
    # --------------------------------------------------------

    if (
        len(widths) >= 2
        and
        PchipInterpolator is not None
    ):

        try:

            (
                surrogate,
                sampled_lower,
                sampled_upper,
            ) = build_surrogate(
                widths,
                frequencies,
            )

            x_plot = np.linspace(
                sampled_lower,
                sampled_upper,
                400,
            )

            y_plot = np.asarray(
                [
                    surrogate(x)
                    for x
                    in x_plot
                ]
            )

            valid = np.isfinite(
                y_plot
            )

            plt.plot(
                x_plot[valid] * 1e6,
                y_plot[valid] / 1e9,
                label="PCHIP surrogate",
            )

        except Exception as exc:

            print(
                f"Could not plot PCHIP: {exc}"
            )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    plt.axhline(
        result[
            "target_frequency"
        ] / 1e9,
        linestyle="--",
        label="Target frequency",
    )

    # --------------------------------------------------------
    # FINAL BEST ACTUAL POINT
    # --------------------------------------------------------

    plt.scatter(
        [
            result[
                "final_cpw_width"
            ] * 1e6
        ],
        [
            result[
                "final_frequency"
            ] / 1e9
        ],
        marker="x",
        s=130,
        label="Best Palace result",
    )

    plt.xlabel(
        "Resonator trace width (µm)"
    )

    plt.ylabel(
        "Frequency (GHz)"
    )

    plt.title(
        "Fixed-Mesh PCHIP + Brent Optimization"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_optimization_summary(
    result,
):
    print(
        "\nOPTIMIZATION SUMMARY"
    )

    print(
        "-" * 60
    )

    print(
        f"Method          : "
        f"{result['method']}"
    )

    print(
        f"Target          : "
        f"{result['target_frequency']/1e9:.9f} GHz"
    )

    print(
        f"Final width     : "
        f"{result['final_cpw_width']*1e6:.9f} µm"
    )

    print(
        f"Final frequency : "
        f"{result['final_frequency']/1e9:.9f} GHz"
    )

    print(
        f"Error           : "
        f"{result['frequency_error']/1e6:.6f} MHz"
    )

    print(
        f"Palace calls    : "
        f"{result['palace_calls']}"
    )

    print(
        f"Converged       : "
        f"{result['converged']}"
    )