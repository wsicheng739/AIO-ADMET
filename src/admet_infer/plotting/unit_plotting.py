import numpy as np


CLS_PROPS = {"bioavailability", "plasma_t_half", "unbound_fraction"}
PROTEIN_MG_ML = 0.5


def _name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _finite(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_classification(property_name):
    return _name(property_name) in CLS_PROPS


def convert_prediction(property_name, task, value):
    if not _finite(value): return {}

    prop, task, value = _name(property_name), _name(task), float(value)

    if prop in CLS_PROPS:
        return {"value": value, "name": "Probability", "unit": ""}

    if prop == "unbound_fraction":
        return {"value": 100 * 10 ** value, "name": "Unbound fraction", "unit": "%"}

    if prop == "solubility":
        return {"value": 10 ** value, "name": "Solubility", "unit": "mol/L"}

    if prop in {"pampa", "mdck_rrck"}:
        return {"value": 1e6 * 10 ** value, "name": "Permeability", "unit": " × 10⁻⁶ cm/s"}

    if prop == "caco2":
        if "ratio" in task:
            return {"value": 10 ** value, "name": "Efflux ratio", "unit": ""}
        return {"value": 1e6 * 10 ** value, "name": "Permeability", "unit": "× 10⁻⁶ cm/s"}

    if prop == "liver_clearance":
        clearance = 10 ** value
        return {
            "value": clearance,
            "name": "CLint",
            "unit": "µL·min⁻¹·mg⁻¹ protein",
            "half_life_min": np.log(2) * 1000 / (clearance * PROTEIN_MG_ML),
        }

    return {}


def format_prediction(property_name, task, value):
    if not _finite(value): return "Prediction: NA"

    prop, task, value = _name(property_name), _name(task), float(value)

    if prop in CLS_PROPS:
        return f"Probability = {value:.3f} ({100 * value:.1f}%)"

    labels = {
        "unbound_fraction": "log10(fu fraction)",
        "solubility": "logS",
        "pampa": "log10(P)",
        "mdck_rrck": "log10(P)",
        "liver_clearance": "log10(CLint)",
    }
    label = "log10(ratio)" if prop == "caco2" and "ratio" in task else labels.get(prop, "log10(P)" if prop == "caco2" else "Prediction")
    model_text = f"{label} = {value:.3f}"

    converted = convert_prediction(prop, task, value)
    if not converted: return model_text

    original_text = f"{converted['name']} = {converted['value']:.3g}"
    if converted["unit"]: original_text += f" {converted['unit']}"
    if "half_life_min" in converted: original_text += f"; microsomal t½ = {converted['half_life_min']:.3g} min"

    return f"{model_text} | {original_text}"


def format_prediction_sd(property_name, value):
    if not _finite(value): return ""

    value = float(value)
    if is_classification(property_name):
        return f"SD = {value:.4f} probability ({100 * value:.2f} percentage points)"
    return f"SD = {value:.4f} log10 units"


def format_prediction_range(property_name, task, value_min, value_max):
    if not _finite(value_min) or not _finite(value_max): return ""

    prop, task = _name(property_name), _name(task)
    value_min, value_max = sorted([float(value_min), float(value_max)])

    if prop in CLS_PROPS:
        return f"Probability range = {100 * value_min:.1f}–{100 * value_max:.1f}%"

    lower = convert_prediction(prop, task, value_min)
    upper = convert_prediction(prop, task, value_max)
    if not lower or not upper: return ""

    text = f"{lower['name']} range: {lower['value']:.3g}–{upper['value']:.3g}"
    if lower["unit"]: text += f" {lower['unit']}"

    if prop == "liver_clearance":
        text += f"; t½ range: {upper['half_life_min']:.3g}–{lower['half_life_min']:.3g} min"

    return text


def attribution_axis_label(property_name, task=""):
    prop, task = _name(property_name), _name(task)

    if prop in CLS_PROPS: return "Segment contribution to predicted probability"

    labels = {
        "unbound_fraction": "Segment contribution to log10(fu fraction)",
        "solubility": "Segment contribution to logS",
        "pampa": "Segment contribution to log10(permeability)",
        "mdck_rrck": "Segment contribution to log10(permeability)",
        "liver_clearance": "Segment contribution to log10(CLint)",
    }

    if prop == "caco2":
        return "Segment contribution to log10(efflux ratio)" if "ratio" in task else "Segment contribution to log10(permeability)"

    return labels.get(prop, "Segment attribution")