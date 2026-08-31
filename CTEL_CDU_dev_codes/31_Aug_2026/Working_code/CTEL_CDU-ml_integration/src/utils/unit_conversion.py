def kelvin_to_celsius(temperature_k: float) -> float:
    """
    Convert temperature from Kelvin to Celsius.

    Parameters
    ----------
    temperature_k : float
        Temperature in Kelvin (must be >= 0)

    Returns
    -------
    float
        Temperature in Celsius
    """
    if temperature_k < 0:
        raise ValueError("Temperature in Kelvin cannot be negative.")

    return temperature_k - 273.15

def celsius_to_kelvin(temperature_c: float) -> float:
    """
    Convert temperature from Celsius to Kelvin.

    Parameters
    ----------
    temperature_c : float
        Temperature in Celsius

    Returns
    -------
    float
        Temperature in Kelvin
    """
    temperature_k = temperature_c + 273.15

    if temperature_k < 0:
        raise ValueError("Resulting temperature in Kelvin cannot be negative.")

    return temperature_k

def g_per_ml_to_kg_per_m3(density_g_ml: float) -> float:
    """
    Convert density from g/mL to kg/m³.

    Parameters
    ----------
    density_g_ml : float
        Density in grams per milliliter

    Returns
    -------
    float
        Density in kilograms per cubic meter
    """
    return density_g_ml * 1000.0


def g_per_ml_to_g_per_cc(density_g_ml: float) -> float:
    """
    Convert density from g/mL to g/cc.

    Note:
    1 mL = 1 cc, so the numerical value remains unchanged.

    Parameters
    ----------
    density_g_ml : float
        Density in grams per milliliter

    Returns
    -------
    float
        Density in grams per cubic centimeter
    """
    return density_g_ml

def tph_to_kg_per_s(flow_tph: float) -> float:
    """
    Convert mass flow rate from ton per hour (TPH) to kg/s.

    Assumption
    ----------
    1 ton = 1000 kg

    Parameters
    ----------
    flow_tph : float
        Mass flow rate in ton per hour

    Returns
    -------
    float
        Mass flow rate in kilograms per second
    """
    return flow_tph * 1000.0 / 3600.0
