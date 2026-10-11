import json
from pathlib import Path

import numpy as np
import pytest

from pythermalcomfort.classes_input import _HeatLimProfiles
from pythermalcomfort.models.heat_lim import Profiles, heat_lim
from tests.conftest import validate_result


def _build_profile(profile_data):
    """Build a HEAT-Lim profile from reference-table data."""
    if profile_data["type"] == "predefined":
        return getattr(Profiles, profile_data["name"])

    if profile_data["type"] == "custom":
        return _HeatLimProfiles(
            mass=profile_data["mass"],
            height=profile_data["height"],
            f_eff=profile_data["f_eff"],
            Icl=profile_data["Icl"],
            Re_cl=profile_data["Re_cl"],
            wmax=profile_data["wmax"],
            srmax=profile_data["srmax"],
        )

    message = f"Unknown profile type: {profile_data['type']}"
    raise ValueError(message)


def test_heat_lim() -> None:
    """Test HEAT-Lim against established V3 reference cases."""
    reference_path = (
        Path(__file__).parent / "data_driven" / "ts_heat_lim.json"
    )

    with reference_path.open(encoding="utf-8") as file:
        reference_table = json.load(file)

    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"].copy()

        inputs["profile"] = _build_profile(inputs["profile"])

        result = heat_lim(**inputs)

        validate_result(
            result,
            entry["outputs"],
            tolerance,
        )
        

def test_heat_lim_arrays():
    """Test vectorized environmental inputs."""
    result = heat_lim(
        tdb=np.array([30, 35, 40]),
        tr=np.array([30, 40, 45]),
        rh=np.array([30, 40, 50]),
        vr=np.array([1, 1, 1]),
        profile=Profiles.YOUNG_ADULT,
    )

    assert np.shape(result.Mmax) == (3,)
    assert np.shape(result.survive6H) == (3,)
    assert np.shape(result.survive3H) == (3,)


def test_heat_lim_broadcasting():
    """Test NumPy broadcasting across environmental inputs."""
    result = heat_lim(
        tdb=np.array([[30], [35]]),
        tr=40,
        rh=np.array([20, 40, 60]),
        vr=1,
        profile=Profiles.YOUNG_ADULT,
    )

    assert np.shape(result.Mmax) == (2, 3)
    assert np.shape(result.survive6H) == (2, 3)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("rh", -1),
        ("rh", 101),
        ("vr", 0),
        ("vr", 16),
    ],
)
def test_heat_lim_invalid_ranges(parameter, value):
    """Reject environmental inputs outside accepted ranges."""
    inputs = {
        "tdb": 35,
        "tr": 40,
        "rh": 30,
        "vr": 1,
        "profile": Profiles.YOUNG_ADULT,
    }

    inputs[parameter] = value

    with pytest.raises(ValueError):
        heat_lim(**inputs)


def test_heat_lim_invalid_profile():
    """Reject invalid personal profile objects."""
    with pytest.raises(TypeError):
        heat_lim(
            tdb=35,
            tr=40,
            rh=30,
            vr=1,
            profile="YOUNG_ADULT",
        )


def test_heat_lim_invalid_broadcasting():
    """Reject environmental inputs with incompatible shapes."""
    with pytest.raises(ValueError):
        heat_lim(
            tdb=np.array([30, 35]),
            tr=np.array([30, 35, 40]),
            rh=30,
            vr=1,
            profile=Profiles.YOUNG_ADULT,
        )