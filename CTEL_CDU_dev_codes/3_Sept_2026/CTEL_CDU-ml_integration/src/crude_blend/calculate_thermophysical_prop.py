from src.utils.core_utility_functions import sigfig
from src.utils.unit_conversion import g_per_ml_to_g_per_cc
class ThermophysicalProperties:
    """
    A class to calculate crude oil thermophysical properties based on
    density, API gravity, sulfur content, and vacuum residue (VR%).
    """

    # -------------------- Constructor --------------------
    def __init__(self):

        # Fixed temperature (assumption)
        self.T_F = 212                 # Fahrenheit
        self.T_C = 5 * (self.T_F - 32) / 9
        self.T_K = self.T_C + 273

    # -------------------- Specific Gravity --------------------
    def compute_specific_gravity(self):
        """
        Computes specific gravity (sg) from API gravity.

        Returns
        -------
        float : specific gravity
        """
        return 141.6 / (131.5 + self.api_gravity)

    # -------------------- Molecular Weight --------------------
    def compute_molecular_weight(self):
        """
        Computes molecular weight using Chen et al. (2002) correlation.

        Returns
        -------
        float : molecular weight (g/mol)
        """
        sg = self.sg
        return (622.5 * sg - 399.57) * (1 + 0.015 * self.vr)

    # -------------------- Specific Heat --------------------
    def compute_specific_heat(self):
        """
        Computes specific heat capacity using a Nelson–Obert-type correlation.
        Output is converted from Btu/lb-F to J/kg-K.

        Returns
        -------
        float : specific heat (J/kg·K)
        """
        Cp_btu = 0.388 + 0.00045 * self.api_gravity + 0.00012 * self.T_F
        Cp_j = Cp_btu * 4186.8 / 2.2046
        return Cp_j

    # -------------------- Thermal Conductivity --------------------
    def compute_thermal_conductivity(self):
        """
        Computes thermal conductivity using an empirical correlation
        adjusted for sulfur and VR%.

        Returns
        -------
        float : thermal conductivity (W/m·K)
        """
        base_k = 0.177 - 0.088 * self.sg + 0.000465 * self.T_K
        correction = 1 - 0.005 * self.sulfur - 0.003 * self.vr
        return base_k * correction

    # -------------------- Viscosity --------------------
    def compute_viscosity(self):
        """
        Computes dynamic viscosity (mu) using:
        1. A kinematic viscosity correlation.
        2. Conversion to dynamic viscosity using density.
        3. Correction for sulfur and VR%.

        Returns
        -------
        float : dynamic viscosity (Pa·s)
        """
        import numpy as np

        log10_nu = 0.799 - 0.602 * np.log10(self.T_F) + 0.29 / self.sg
        nu = 10 ** log10_nu  # kinematic viscosity (cSt)

        # Convert to dynamic viscosity:
        #   μ = ν * density(kg/m3)
        # density g/ml -> g/cc
        density_g_cc = g_per_ml_to_g_per_cc(self.density)
        # density input is in g/cc → convert to kg/m3
        mu = nu * (density_g_cc * 1000) / 1e6

        correction = (1 + 0.03 * self.vr) * (1 + 0.05 * self.sulfur)
        return mu * correction

    def get_thermophysical_properties(self, density, api_gravity, sulfur, vr):
        """
        Parameters
            ----------
            density : float
                Density of the crude oil (g/ml).
            api_gravity : float
                API Gravity of the crude oil.
            sulfur : float
                Sulfur content in weight percent (%).
            vr : float
                Vacuum residue (VR) content in weight percent (%).
        
            Assumptions
            -----------
            1. Temperature for calculations is fixed at:
                - T_F = 212 °F  (≈ 100 °C)
                - T_C = 100 °C
                - T_K = 373 K
            2. Correlation equations are based on:
                - Chen et al. (2002) for molecular weight.
                - Nelson–Obert type correlation for specific heat.
                - Empirical correlations for thermal conductivity & viscosity.
            3. All inputs must be provided as single numeric values.
        
            Attributes (Outputs)
            --------------------
            MW : float
                Estimated molecular weight (g/mol).
            Cp : float
                Specific heat capacity (J/kg·K).
            k : float
                Thermal conductivity (W/m·K).
            mu : float
                Dynamic viscosity (Pa·s).
            sg : float
                Specific gravity (dimensionless).
        """

        self.density = density
        self.api_gravity = api_gravity
        self.sulfur = sulfur
        self.vr = vr


        # Compute all properties on initialization
        self.sg = sigfig(self.compute_specific_gravity())
        self.MW = sigfig(self.compute_molecular_weight())
        self.Cp = sigfig(self.compute_specific_heat())
        self.k = sigfig(self.compute_thermal_conductivity())
        self.mu = sigfig(self.compute_viscosity())

        thermo_props = {
            "Specific Gravity": self.sg,
            "Molecular Weight": self.MW,
            "Specific Heat Cp": self.Cp,
            "Thermal Conductivity": self.k,
            "Viscosity mu": self.mu,
        }

        return thermo_props
    
if __name__ == "__main__":
    sample = ThermophysicalProperties()

    density=0.85,     # g/cc
    api_gravity=35,   # API
    sulfur=1.2,       # %
    vr=10             # %

    thermo_props = sample.get_thermophysical_properties(density, api_gravity, sulfur, vr)
    print("Specific Gravity:", thermo_props["Specific Gravity"])
    print("Molecular Weight:", thermo_props["Molecular Weight"])
    print("Specific Heat Cp:", thermo_props["Specific Heat Cp"])
    print("Thermal Conductivity:", thermo_props["Thermal Conductivity"])
    print("Viscosity mu:", thermo_props["Viscosity mu"])
