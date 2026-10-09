import numpy as np
from dataclasses import dataclass
from pythermalcomfort.classes_input import HeatLimInputs, _HeatLimProfiles
from pythermalcomfort.classes_return import HEATLim

Lh_vap =  2426	#J·g-1	Heat of vaporisation of sweat at 30⁰C. 
σ = 5.67E-08	#W·m-2·K-4	Stefan-Boltzmann constant.
LR	= 16.5	#K·kPa-1	Lewis Relation.


@dataclass(frozen=True)
class Profiles:
    """Namespace of predefined HEAT-Lim personal profiles.

Use attributes such as `Profiles.OLDER_ADULT` or `Profiles.YOUNG_ADULT` to
access reference profile values.

NOTE: Morfology values defer from Vanos et. al. (2023), and the values are representative
 of the US population between 2015 and 2018 (Fryar et al., 2021). Corresponding to morphology 
 for or young (20-29) and older (60-69) adults females.
 
We recommend selecting body morphology values appropriate to the population of interest.
 These values may be obtained from nationally representative anthropometric datasets or from references to healthy populations.

"""

    YOUNG_ADULT = _HeatLimProfiles(mass=74.9, height=1.626, f_eff =0.73, Icl = 0.36, 
                              Re_cl = 0.01, wmax = 0.75, srmax =0.41)
    OLDER_ADULT = _HeatLimProfiles(mass=78.2, height=1.607, f_eff =0.73, Icl = 0.36, 
                             Re_cl = 0.01, wmax = 0.52, srmax =0.33)
    
    

def Psa_kPa_from_TaC(T_C:float) -> float:
    """Estimate saturated vapour pressure from air temperature.

Parameters
----------
T_C : float
    Dry-bulb air temperature [°C]

Returns
-------
float
    Saturated vapour pressure [kPa].
"""
    
    Psa_kPa = np.exp(18.956-(4030.18/(T_C+235)))/10
    return Psa_kPa


def Pv_kPa_from_Psa_RH(Psa_kPa:float,RH:float) -> float:
    """Estimate vapour pressure from saturated vapour pressure and relative humidity.

Parameters
----------
Psa_kPa : float
    Saturated vapour pressure [kPa].
RH : float
    Relative humidity [%].

Returns
-------
float
    Vapour pressure [kPa].
"""
    Pv_kPa = Psa_kPa*RH/100
    return Pv_kPa

def get_METadults_conversion_factor(RER) -> float:
    """
    Calculates energy expenditure conversion factor in METx to W/kg  based on 
    the basal metabolic rate (BMR) across life span.    
    Approaches are based on the BMR reported on the compendiums of physical activity,
    for adults older than 18 with fixed (VO2) and Respiratory Exchange Ratio (RER).    
   
    Standardized resting metabolic rate definition 3.5 ml O2.kg⁻¹.min⁻¹ (Jetté et al., 1990)
    for adults 18-60, with a fixed energy yield of 20.38 kJ/L O₂ (i.e. respiratory exchange ratio of ~0.85).
    
    V1: Vanos et al.2023 assumes a respiratory exchange ratio of ~0.97. Thus a fixed energy yield of 21 kJ/L O₂ . 
    (See Cramer and Jay, 2019).
    
    Notice, future developments will add variations for kids and older adults based on Butte, et al. (2017) and Willis, et al. (2024) respectively.
    Also, Harris-Benedict RMR equations could be used to reflect more  accurate inter-individual variations
    in Age, sex, height, and weight in METx estimates (Hermman, 2024), once more data is available
    for a comprenhensive validation of inter-individual physical activity intensity across life span in the heat.
    
    Bibliography:
        Compendiums of physical activity for the youth (Butte et al., 2017)
        Compendiums of physical activity for adults 18-40 (Hermman et al., 2024)
        Compendiums of physical activity for adults 60+ (Willis et al., 2024)
        
Parameters:
----------
RER: float
    Respiratory Exchange Ratio.

Returns:
----------
float
    The conversion factor for MET (young adults BMR) in Watts per kilogram (W*kg^-1)
    """
    
    #Young adults estimates
    Vo2_bmr = 3.5 #3.5 ml O2.kg⁻¹.min⁻¹ (Jetté et al., 1990)
    # Energy yield in kJ/L of O2 as approximation in Cramer and Jay, when C02 production is not sampled
    # Note if the data was in kcal/L would need to conver to kJ/L first with (1 kcal = 4.186 kJ)
    if RER == 0.97:
        kJ_per_liter = 21 #kJ/l O2
    elif RER == 0.85:
        kJ_per_liter = 20.38 #kJ/l O2
        
    #Option 1: 21 kJ/l O2 (equivalent to a RER of ~0.97) ---- roughly 90 % carbs, 10% fat
    #Option 2: 20.38 kJ/l O2 (equivalent toa RER of ~0.85) ---- roughly 50% carbs, 10 % fat
    
    # Convert kJ/L to Joules/mL (multiplied by 1000 for Joules, divided by 1000 for mL, cancels out)
    joules_per_ml = kJ_per_liter
    
    #Calculate total energy expenditure per minute (J/kg/min)
    # Vo2_bmr is the oxigen expenditure at rest and joules_per_ml say how much heat is produced by that oxygen comsuption
    joules_per_kg_per_min = Vo2_bmr * joules_per_ml
    
    # Convert minutes to seconds (Joules/sec = Watts) -> W/kg
    watts_per_kg = joules_per_kg_per_min / 60.0
    return round(watts_per_kg, 3)
   

factor_MET_YOUNG_ADULT_OLD = get_METadults_conversion_factor(0.97) #V1/V2
# print(f"1 MET 18-40 OLD= {factor_MET_YOUNG_ADULT_OLD} W*kg^-1")
factor_MET_YOUNG_ADULT_NEW = get_METadults_conversion_factor(0.85) #V3

def MET_to_MetabolicRate_W_Mass(METS:float, Mass:float) -> float:
    """Convert metabolic heat production from metabolic equivalents to watts.
    
    The conversion factor for metabolic heat production is based on
    standardized resting metabolic rate definition 3.5 ml O2.kg⁻¹.min⁻¹ (Jetté et al., 1990)
    with a fixed energy yield of 21 kJ.l O2⁻¹ (i.e. respiratory exchange ratio of ~0.97). 
    (See Cramer and Jay, 2019)
    The combination of both factors leads a conversion factor of 1 MET=1.225 W/kg as a result.

----------
METS : float
    Energy expenditure [MET].
Mass : float
    Body mass [kg].

Returns
-------
float
    Metabolic power [W].
"""
    Watts = METS*factor_MET_YOUNG_ADULT_NEW*Mass
    return Watts


def MetabolicRate_W_to_MET_Mass(Watts:float, Mass:float) -> float:
    """ Convert metabolic heat production from watts to metabolic equivalents.
    
    The conversion factor for metabolic heat production is based on
    standardized resting metabolic rate definition 3.5 ml O2.kg⁻¹.min⁻¹ (Jetté et al., 1990)
    with a fixed energy yield of 21 kJ.l O2⁻¹ (i.e. respiratory exchange ratio of ~0.97). 
    (See Cramer and Jay, 2019)
    The combination of both factors leads a conversion factor of 1 MET=1.225 W/kg as a result.
        

Parameters
----------
Watts : float
    Metabolic power [W].
Mass : float
    Body mass [kg].

Returns
-------
float
    Energy expenditure [MET].
"""

    METS = Watts/(factor_MET_YOUNG_ADULT_NEW*Mass)
    return METS

def AD_from_mass_height(Mass:float,Height:float) -> float:  
    """Estimate Dubois body surface area from mass and height.

Parameters
----------
Mass : float
    Body mass [kg].
Height : float
    Body height [m].

Returns
-------
float
    Body surface area [m²].
"""
    ad = 0.202*(Mass**0.425)*(Height**0.725)
    return ad


# =============================================================================
# Dry Heat Loss from the Skin (C+R)		
# =============================================================================

def fcl_from_Icl(Icl:float) -> float:
    """Estimate clothing area factor from clothing insulation.

Parameters
----------
Icl : float
    Clothing insulation [clo].

Returns
-------
float
    Clothing area factor [dimensionless].
"""
    fcl = 1 + (0.31*Icl)
    return fcl

def hc_cof_from_Av(Av_ms:float) -> float:
    """Estimate convective heat transfer coefficient from air speed.
    
    Uses Parsons, K. (2014). Human Thermal Environments (Third edit). CRC Press. 
    https://doi.org/10.1201/b16750 (page 51)

Parameters
----------
Av_ms : float
    Air speed [m·s⁻¹].

Returns
-------
float
    Convective heat transfer coefficient [W·m⁻²·K⁻¹].
"""
    hc_cof = np.where(Av_ms < 0.2, 3.61,((Av_ms**0.6)*8.3))
    return hc_cof

def Rcl_from_Icl(Icl:float) -> float:
    """Estimate dry clothing heat resistance from clothing insulation.

Parameters
----------
Icl : float
    Clothing insulation [clo].

Returns
-------
float
    Dry clothing heat resistance [m²·°C·W⁻¹].
"""
    #Intrinsic clothing insulation
    Rcl = 0.155*Icl
    return Rcl
        
def hr_cof_from_radiant_features(Tr_C:float,Tsk_C:float,Emm_sk:float,f_eff:float) -> float:
    """Estimate radiative heat transfer coefficient.

Parameters
----------
Tr_C : float
    Mean radiant temperature [°C].
Tsk_C : float
    Mean skin temperature [°C].
Emm_sk : float
    Area-weighted emissivity of the clothed body surface [dimensionless].
f_eff : float
    Effective radiative area factor [dimensionless].

Returns
-------
float
    Radiative heat transfer coefficient [W·m⁻²·K⁻¹].
"""
    Boltzmann = 5.67*10**-8 # [W·m-2·K-4]	Stefan-Boltzmann constant.
    hr_cof = 4*Emm_sk*Boltzmann*f_eff*((273.2+(Tsk_C+Tr_C)/2)**3)

    return hr_cof
    

def h_coef_from_hc_hr(hc_cof:float,hr_cof:float) -> float:
    """Estimate combined convective and radiative heat transfer coefficient.

Parameters
----------
hc_cof : float
    Convective heat transfer coefficient [W·m⁻²·K⁻¹].
hr_cof : float
    Radiative heat transfer coefficient [W·m⁻²·K⁻¹].

Returns
-------
float
    Combined heat transfer coefficient [W·m⁻²·K⁻¹].
"""
    h_cof = hc_cof + hr_cof
    return h_cof

def to_from_hr_tr_hc_ta(hr_cof:float,mrt_C:float,hc_cof:float,Ta_C:float) -> float:
    """Estimate operative temperature from radiative and convective coefficients.

Parameters
----------
hr_cof : float
    Radiative heat transfer coefficient [W·m⁻²·K⁻¹].
mrt_C : float
    Mean radiant temperature [°C].
hc_cof : float
    Convective heat transfer coefficient [W·m⁻²·K⁻¹].
Ta_C : float
    Ambient temperature [°C].

Returns
-------
float
    Operative temperature [°C].
"""
    to_C=((hr_cof*mrt_C)+(hc_cof*Ta_C))/(hr_cof+hc_cof)
    return to_C

def Dry_Heat_Loss_c_plus_r(Tsk_C:float,to_C:float,Icl:float,h_cof:float,Ad:float) -> float:
    """Estimate dry heat loss by convection and radiation.

Parameters
----------
Tsk_C : float
    Mean skin temperature [°C].
to_C : float
    Operative temperature [°C].
Icl : float
    Clothing insulation [clo].
h_cof : float
    Combined heat transfer coefficient [W·m⁻²·K⁻¹].
Ad : float
    Body surface area [m²].

Returns
-------
float
    Dry heat loss [W].
"""
    fcl	= fcl_from_Icl(Icl)	#[Dimensionless]	Clothing area factor. Ratio of clothed body surface to nude body surface.
    Rcl	= Rcl_from_Icl(Icl)	#[m2·⁰C·W-1]	Intrinsic clothing insulation
    
    
    Dry_Heat_Loss_AD = (Tsk_C-to_C)/(Rcl+(1/(h_cof*fcl)))
    Dry_Heat_Loss = Dry_Heat_Loss_AD*Ad #Conversion from w.M-2 to W
    return Dry_Heat_Loss
    

# =============================================================================
# Heat Loss from Respiration (Cres+Eres)	
# =============================================================================

	
def Cres_from_M_Ta(M:float,Ta_C:float,Ad:float) -> float:
    """Estimate respiratory convective heat loss.

Parameters
----------
M : float
     Rate of metabolic energy expenditure [W].
Ta_C : float
    Dry-bulb air temperature [°C].
Ad : float
    Body surface area [m²].

Returns
-------
float
    Respiratory convective heat loss [W].
"""
    Cres = 0.0014*M*(34-Ta_C)*Ad
    return Cres

def Eres_from_M_Pa(M:float,Pa_kPa:float,Ad:float) -> float:
    """Estimate respiratory evaporative heat loss.

Parameters
----------
M : float
     Rate of metabolic energy expenditure [W].
Pa_kPa : float
    Ambient vapour pressure [kPa].
Ad : float
    Body surface area [m²].

Returns
-------
float
    Respiratory evaporative heat loss [W].
"""
    Eres = 0.0173*M*(5.86618428-Pa_kPa)*Ad
    
    return Eres


# =============================================================================
# Estimation of Evaporation required and ambiental possible
# =============================================================================


def Ereq(M:float,W:float,Dry_Heat_Loss:float,Cres:float,Eres:float) -> float:
   
    """Estimate required evaporative heat loss for heat balance.

Parameters
----------
M : float
    Rate of metabolic energy expenditure [W].
W : float
    External work [W].
Dry_Heat_Loss : float
    Dry heat loss by convection and radiation [W].
Cres : float
    Respiratory convective heat loss [W].
Eres : float
    Respiratory evaporative heat loss [W].

Returns
-------
float
    Required evaporative heat exchange at skin to attain heat balance [W].
"""
    
    Hprod = M - W
    CEplus_res = Cres + Eres
    Ereq = Hprod  - Dry_Heat_Loss - CEplus_res
    
        
    return Ereq

def he_cof(hc_cof:float) -> float:
    """Estimate evaporative heat transfer coefficient using the Lewis relation.

Parameters
----------
hc_cof : float
    Convective heat transfer coefficient [W·m⁻²·K⁻¹].

Returns
-------
float
    Evaporative heat transfer coefficient [W·m⁻²·kPa⁻¹].
"""  
    he_cof =hc_cof*LR
    return he_cof



def Emax_env(Psk_s:float,Pv_kPa:float,Re_cl:float,he_cof:float,Icl:float,AD:float) -> float:
    """Estimate maximum evaporative heat loss at skin constrained by clothing.
    
    This is  the maximum evaporative heat loss for a given thermal environment 
    and clothing, also known as the biophysical evaporative heat loss.

Parameters
----------
Psk_s : float
    Saturated vapour pressure at the skin [kPa].
Pv_kPa : float
    Ambient vapour pressure [kPa].
Re_cl : float
    Evaporative clothing resistance [m²·kPa·W⁻¹].
he_cof : float
    Evaporative heat transfer coefficient [W·m⁻²·kPa⁻¹].
Icl : float
    Clothing insulation [clo].
AD : float
    Body surface area [m²].

Returns
-------
float
    Maximum evaporative heat loss constrained by clothing [W].
"""
    fcl	= fcl_from_Icl(Icl)	#Ratio of clothed body surface to nude body surface (dimensionless)

    Emax_env_AD = (Psk_s-Pv_kPa)/(Re_cl+(1/(he_cof*fcl)))            
    Emax_env = Emax_env_AD*AD 
    return Emax_env

def Emax_wettedness(wmax:float,Psk_s:float,Pv_kPa:float,Re_cl:float,he_cof:float,Icl:float,AD:float) -> float:
    """Estimate maximum (environmental allowed) evaporative heat loss constrained by skin wettedness and clothing.
    
    This function estimates the biophysical evaporative heat loss accounting as additional 
    constraing the capacity to physiologically wet the skin and thus distribute sweat across 
    the skin surface in humid environments. This restriction is applied using the maximum skin wettedness.

Parameters
----------
wmax : float
    Maximum skin wettedness [dimensionless].
Psk_s : float
    Saturated vapour pressure at the skin [kPa].
Pv_kPa : float
    Ambient vapour pressure [kPa].
Re_cl : float
    Evaporative clothing resistance [m²·kPa·W⁻¹].
he_cof : float
    Evaporative heat transfer coefficient [W·m⁻²·kPa⁻¹].
Icl : float
    Clothing insulation [clo].
AD : float
    Body surface area [m²].

Returns
-------
float
    Maximum evaporative heat loss constrained by wettedness [W].
"""
    
    fcl	= fcl_from_Icl(Icl)	#Ratio of clothed body surface to nude body surface (dimensionless)

    Emax_wettedness_AD = wmax*(Psk_s-Pv_kPa)/(Re_cl+(1/(he_cof*fcl)))            
    Emax_wettedness = Emax_wettedness_AD*AD 
    return Emax_wettedness


    
def Emax_sweat_rate(srmax:float,Lh_vap:float, density:float, r:float) -> float:
    """Estimate maximum evaporative heat loss given maximum sweat rate.
    
This function estimates the evaporative heat loss after from the maximum volume 
of sweat that people can release based on their maximum hourly sweat rate (SRmax)
and accounting sweat efficiency. (r)

In this equation 3.6 is the conversion factor to account the time from hours 
to seconds and the volume of sweat to mass, assuming that sweat density is similar like water

This metric implies a constant sweat rate and a completely water replenish.

Parameters
----------
srmax : float
    Maximum sweat rate [L·h⁻¹].
Lh_vap : float
    Latent heat of vaporisation of sweat [J·g⁻¹].
density : float
    Sweat density [dimensionless].
r : float
    Sweating efficiency [dimensionless].

Returns
-------
float
    Maximum evaporative heat loss constrained by sweat rate [W].
"""
    Emax = ((srmax*Lh_vap*density)/3.6)*r
    return Emax
# =============================================================================
# Body capacity to sweat
# =============================================================================

#the body is capable? how stress the body is?

def wreq_HSI_skin_wettedness(Ereq:float,Emax_env:float) -> float:
    """Estimate required skin wettedness for heat balance.

Parameters
----------
Ereq : float
    Required evaporative heat loss at skin to attain heat balance [W].
Emax_env : float
    Maximum environmental evaporative heat loss constrained by clothing [W].

Returns
-------
float
    Required skin wettedness [dimensionless].
"""
    with np.errstate(divide="ignore", invalid="ignore"):
        wreq = np.divide(Ereq, Emax_env, out=np.full_like(Ereq, np.nan, dtype=float), where=Emax_env != 0)
    return wreq		
	

def Sweating_efficiency_r(wreq:float) -> float:
    """Estimate sweating efficiency from required skin wettedness.

Parameters
----------
wreq : float
    Required skin wettedness [dimensionless].

Returns
-------
float
    Sweating efficiency [dimensionless].
"""

    r = np.where(wreq<1, 1-((wreq**2)/2),0.5)
    r = np.where(r>1,1,r)
    return r

def SRreq(Ereq:float,r:float,Lh_vap:float) -> float:
    """Estimate required sweat rate for heat balance.

    In this equation 3.6 is the conversion factor to account the time from hours 
    to seconds and the volume of sweat to mass, assuming that sweat density like
    water density.
    
    Cramer and jay Eq.21 
    Note that for shorter exercise durations this method will overestimate whole body sweat rate,
    as non-steady-state sweat rates during the earlier stages of  exercise are lower.

Parameters
----------
Ereq : float
    Required evaporative heat loss [W].
r : float
    Sweating efficiency [dimensionless].
Lh_vap : float
    Latent heat of vaporisation of sweat [J·g⁻¹].

Returns
-------
float
    Required sweat rate [L·h⁻¹].
"""
    SRreq =((Ereq/r)/Lh_vap)*3.6
    return SRreq
   
# =============================================================================
# Survivability and livability core functions 
# =============================================================================

def Ssurvive_from_Exposure_time(Exp_time) -> float:
    """Estimate the critical heat storage rate for the selected exposure time.
    
This function estimates the critical rate of heat storage (Ssurv) before inevitable
heat stroke death during rest for exposure times of 6 and 3 hours.
Here is assumed to have a linear increase of 6°C (Starting temperature at 37 °C)
and a Human body-specific heat capacity of C_p= 2.98 kJ/(Kg.Â°C). 

Cp: Xiaojiang Xu, Timothy P. Rioux & Michael P. Castellani (2022) The specific heat of the human 
body is lower than previously believed: The Journal Temperature toolbox, Temperature, 
DOI: 10.1080/23328940.2022.2088034

Parameters
----------
Exp_time : float
    Exposure time [h]. Valid values are 1, 3, or 6.

Returns
-------
float
    Critical heat storage rate [W·kg⁻¹].
"""
    valid = {1, 3, 6}
    if Exp_time not in valid:
        raise ValueError("results: status must be one of %r." % valid)
    #Estimation of constant power along exposure time
    DQ = 2.98*6  #CP.dT (change of temperature of 6 degrees)
    Ssurvive = (DQ * 1000)/(Exp_time*3600) #1000 is conversion from KJ to J, and 3600 conversion seg to hours
    
    return Ssurvive

def Survivability(Exp_time:float,Ereq:float,Emax_wettedness:float,Emax_sweat:float,Sreq:float,SRmax_abs:float,Mass:float) -> tuple:
    """Assess survivability for a given exposure time.

This function assesses human survivability under resting conditions by determining whether a person can avoid a lethal core temperature rise during 3-hour or 6-hour exposures to a constant thermal environment. The critical rate of heat storage (Ssurvive) associated with each exposure duration is estimated by the Ssurvive_from_Exposure_time() function within this module.

This assessment recognizes that a person may survive despite experiencing uncompensable heat stress. In such cases, heat storage may be positive and body core temperature may continue to rise, provided it does not exceed the assumed lethal threshold of 43°C within the specified exposure duration.

See Vanos et al. (2023) for the complete environmental and physiological assumptions underlying the model.

In addition to determining survivability as a Boolean outcome, the algorithm assigns a categorical flag describing the physiological mechanism that limits survivability (see Supplemental Material, Figure S1: Model workflows for survivability and livability).

The survivability categories are defined as follows:

1: Survives while remaining within maximal sweating limits.
2: Survives despite exceeding maximal sweat production capacity, because the resulting evaporative heat loss remains sufficient
   to keep body heat storage below the lethal threshold.
3: Does not remain thermally compensable because environmental evaporative limitations (e.g., high humidity) are restrictive,
   but the resulting heat storage remains below the lethal threshold.
4: Does not survive because maximum sweat production is insufficient to dissipate the required evaporative heat loss, despite sufficient
   environmental evaporative capacity.
5: Does not survive because both environmental evaporative limitations and maximum sweat production capacity are insufficient.

Parameters
----------
Exp_time : float
    Exposure time [h].
Ereq : float
    Required evaporative heat loss [W].
Emax_wettedness : float
    Maximum evaporative heat loss constrained by clothing and maximum skin wettedness [W].
Emax_sweat : float
    Maximum evaporative heat loss allowable by maximum sweat rate [W].
Sreq : float
    Required sweat rate [L·h⁻¹].
SRmax_abs : float
    Maximum sweat rate (non surface-area normalized) [L·h⁻¹].
Mass : float
    Body mass [kg].

Returns
-------
tuple
    Boolean survivability flag and survivability class flag.
"""
    valid = {3, 6}
    if Exp_time not in valid:
        raise ValueError("results: status must be one of %r." % valid)

    Ssurvive = Ssurvive_from_Exposure_time(Exp_time)

    # Boolean criteria
    condition1 = (Ereq - Emax_wettedness) <= (Ssurvive * Mass)
    condition21 = Sreq <= SRmax_abs
    condition22 = (Ereq - Emax_sweat) <= (Ssurvive * Mass)
    
    # Survivability decision
    survivability = np.where(
        condition1,
        np.where(condition21, True, condition22),
        False,
    )
    
    # Survivability class flag
    flag_survivability = np.where(
        ~condition1,
        np.where(~condition22, 5, 3),
        np.where(
            condition21,
            1,
            np.where(condition22, 2, 4),
        ),
    )
    
    if np.ndim(survivability) == 0:
        survivability = survivability.item()
    
    if np.ndim(flag_survivability) == 0:
        flag_survivability = flag_survivability.item()
    
    return survivability, flag_survivability


def livability_Mmax(Ereq:float,Emax_wettedness:float,Emax_sweat:float,M_rest:float, Mass:float) -> float:
    """Estimate livability to a given environment.

Livability is the maximum metabolic rate (Mmax) that can be generated before S≥0, 
or sustained compensable heat stress, whith M = Hprod. The Mmax value indicates the 
sustained activity levels (intensity but not duration) possible without unchecked 
rises in Tcore (i.e., uncompensable heat stress) within a given steady-state 
thermal environment. 

After checking that heat stress is compensable (Ereq≤Emaxlim) Mmax is estimated 
as follows.

    M_max  = Emax_lim  -  H_loss  
    or
    M_max  = Emax_lim  -  H_loss
    
Notice that:
    - Emax_lim is the minimum value (or most restrictive) among Emax_wettedness and Emax_sweat
    - H_loss can be represented either by (Hdry +Cres +Eres) or (Ereq + Hprod). 

Mmax is capped at 10 METs to prevent unrealistically high estimates under 
cooler conditions and removing the previous recommendation to restrict livability
analyses to air temperatures above 25°C. This change was already applied in HEAT-Lim V2.
    
Parameters
----------
Ereq : float
    Required evaporative heat loss [W].
Emax_wettedness : float
    Maximum evaporative heat loss constrained by wettedness [W].
Emax_sweat : float
    Maximum evaporative heat loss constrained by sweat rate [W].
M_rest : float
    Rest-to-sedentary metabolic rate [W].
Mass : float
    Body mass [kg].

Returns
-------
tuple
    
float  
    Maximum safe internal heat production in [Wats],
    that a person can generate without a sustained positive rate of heat storage
    in the prevailing environment.
    This estimate is not capped at 10 METs, then this value at the momment can be really high upon cold conditions.
float  
    Capped maximum metabolic rate [MET], translated into relevant physical activity metric with reference to young adults compendium

"""

    Emax_lim = np.min([Emax_wettedness,Emax_sweat],axis=0)
    compensability = np.where(Ereq <= Emax_lim, True,False) 
    
    Mmax = Emax_lim - Ereq + M_rest
    
    Mmax_W = np.where(compensability, Mmax, np.nan) # This set the minimum value at rest-sedentary as 1.5 METs
    Mmax_MET = MetabolicRate_W_to_MET_Mass(Mmax_W,Mass) #Convert to METs
    Mmax_capped = np.where(Mmax_MET > 10, 10, Mmax_MET) #if Mmax is greather than 10 METs
    return Mmax_W, Mmax_capped


def non_livable (Mmax) -> bool:
    '''This function indicates wether an environment is un-livable because is non-compensable at 1.5 METs.

When Mmax is nan mean heat stress is non-compensable for sedentary-to-light activities and was set as a non-value.
In such situation, a person can survive but is not able to live (no activity is possible without storage heat internally).

This situation is dangerous if sustained over time and that is why a third state: 
"un-livable" is created to account for those exposures in which a person could survive under
uncompensable heat stress but for limited time.

An upper limit for this condition can be set with the survivability assesment (for future developments).'''
    
    non_livable = np.isnan(Mmax)
    return non_livable
    


# =============================================================================
# Full HEAT-Lim process
# =============================================================================

def heat_lim(
    # tdb: NumericInput | np.ndarray, (Commented to be uncommented for pythermalcomfort)
    # tr: NumericInput | np.ndarray,
    # rh: NumericInput | np.ndarray,
    # vr: NumericInput | np.ndarray,
    tdb: float | np.ndarray,
    tr: float | np.ndarray,
    rh: float | np.ndarray,
    vr: float | np.ndarray,
    profile: _HeatLimProfiles,
    ) -> HEATLim:
    """Assess human survivability and livability under constant environmental conditions.

ASU-USYD HEAT-Lim applies partitional calorimetry principles to estimate human-environment heat exchanges and determine whether a person survives a 3-hour or 6-hour exposure, as well as the maximum sustainable metabolic rate (`Mmax`) under the same conditions. The model is intended for steady-state analyses in which thermoregulatory responses have reached their physiological limits, including maximum skin temperature, maximal sweat rate, and maximum skin wettedness.

The formulation follows the original HEAT-Lim framework described by Vanos et al. (2023), with updates introduced in HEAT-Lim V3 to improve the consistency of the model workflow and integration with the `pythermalcomfort` package.

## Parameters

tdb : float or array-like
    Dry-bulb air temperature [°C].
tr : float or array-like
    Mean radiant temperature [°C].
rh : float or array-like
    Relative humidity [%].
vr : float or array-like
    Relative air speed [m·s⁻¹].
profile : _HeatLimProfiles
    Profile-specific dataclass with body morphology and clothing properties.

## Returns

HEATLim 
Dataclass containing survivability and livability `Mmax` metrics.

## Examples

Scalar input:

.. code-block:: python

```
from HEATLim_V3 import heat_lim, Profiles

results = heat_lim(
    tdb=38,
    tr=45,
    rh=20,
    vr=0.8,
    profile=Profiles.YOUNG_ADULT,
)

print(f"Mmax: {results.Mmax:.2f} METs")
print(f"Survive 3H constant exposure: {results.survive3H}")
```

Array input:

.. code-block:: python

```
import numpy as np
from HEATLim_V3 import heat_lim, Profiles

temperature = np.arange(25, 60, 0.1)
humidity = np.arange(0.5, 100.5, 0.5)
Ta_C, RH = np.meshgrid(temperature, humidity)
Av_ms = np.ones(Ta_C.shape) * 1
mrt_C = Ta_C.copy()

results = heat_lim(
    tdb=Ta_C,
    tr=mrt_C,
    rh=RH,
    vr=Av_ms,
    profile=Profiles.YOUNG_ADULT,
)

print(results.Mmax.shape)
print(results.survive3H.shape)
print(results.Mmax[0, 0])
print(results.survive3H[0, 0])
```
"""

    inputs = HeatLimInputs(
        tdb=tdb,
        tr=tr,
        rh=rh,
        vr=vr,
        profile=profile,
    )

    tdb = np.asarray(inputs.tdb, dtype=float)
    tr = np.asarray(inputs.tr, dtype=float)
    rh = np.asarray(inputs.rh, dtype=float)
    vr = np.asarray(inputs.vr, dtype=float)
    profile = inputs.profile
    
    #---------------------------------------------------------------------------
    #   Setting personal profile, clothing and activity / utility constants
    #---------------------------------------------------------------------------
    
    mass = profile.mass #[kg] Body mass
    AD  = AD_from_mass_height(profile.mass,profile.height) #[m²] Body surface area
    wmax = profile.wmax #[dimensionless] Maximum skin wettedness
    srmax_rel = profile.srmax #[l·h⁻¹.m⁻²] Maximum sweat rate relative to surface area
    srmax_abs = srmax_rel*AD #[l·h⁻¹] Maximum sweat rate 
    f_eff	= profile.f_eff #[dimensionless] Effective radiative area factor (Relative to corporal posture)
    Icl = profile.Icl #[clo] Clothing insulation
    Re_cl = profile.Re_cl#[m²·kPa·W⁻¹] Evaporative clothing resistance
    Tskmax_C = 36  # [°C] Maximum skin temperature (Informed by Fisher et al. (2025)).
    Emm_sk = 0.98 # [dimensionless] Emissivity of human skin
    
    #Estimations
    Psa_kPa = Psa_kPa_from_TaC(tdb)  # [kPa] Saturated vapour pressure.
    Pv_kPa = Pv_kPa_from_Psa_RH(Psa_kPa, rh)  # [kPa] Ambient vapour pressure.
    
    # =============================================================================
    #  Metabolic Heat Flux (Hprod = M-W)	
    # =============================================================================
    M_rest = MET_to_MetabolicRate_W_Mass(1.5, mass) # [W] Resting metabolic power assuming 1.5 METs (~1.8 W·kg⁻¹).
    M = M_rest
    W =  0 #no work done.						            		
    # =============================================================================
    # Dry Heat Loss from the Skin (C+R)		
    # =============================================================================
    hc_cof = hc_cof_from_Av(vr)  # [W·m⁻²·K⁻¹] Convective heat transfer coefficient.
    hr_cof = hr_cof_from_radiant_features(tr, Tskmax_C, Emm_sk, f_eff)  # [W·m⁻²·K⁻¹] Linear radiative heat transfer coefficient.
    h_cof = h_coef_from_hc_hr(hc_cof, hr_cof)  # [W·m⁻²·K⁻¹] Combined heat transfer coefficient.
    to_C = to_from_hr_tr_hc_ta(hr_cof, tr, hc_cof, tdb)  # [°C] Operative temperature.
    Dry_Heat_Loss = Dry_Heat_Loss_c_plus_r(Tskmax_C, to_C, Icl, h_cof, AD)  # [W] Dry heat loss by convection and radiation.
    # =============================================================================
    #  Heat Loss from Respiration (Cres+Eres)	
    # =============================================================================
    Cres = Cres_from_M_Ta(M, tdb, AD)  # [W] Respiratory convective heat loss.
    Eres = Eres_from_M_Pa(M, Pv_kPa, AD)  # [W] Respiratory evaporative heat loss.
    # =============================================================================
    # Estimation of Evaporation required and ambiental possible
    # =============================================================================
    #  evaporation required for heat balance per unit area.
    Ereq_aux = Ereq(M, W, Dry_Heat_Loss, Cres, Eres)  # [W] Required evaporative heat loss for survivability.
    # =============================================================================
    # Estimation  Byophisical (Env + clothing) and accouting wettedness(wettedness)
    # maximum evaporative heat loss
    # =============================================================================
    Psk_s = Psa_kPa_from_TaC(Tskmax_C)  # [kPa] Saturated vapour pressure at the skin. Assumption of saturated water RH=100% at skin surface.
    he_cof_aux = he_cof(hc_cof)  # [W·m⁻²·kPa⁻¹] Evaporative heat transfer coefficient.
    # Byophisical Emax (Env + clothing) 
    Emax_env_aux = Emax_env(Psk_s, Pv_kPa, Re_cl, he_cof_aux, Icl, AD)  # [W] Maximum evaporative heat loss constrained by clothing.
    Emax_env_aux = np.maximum(Emax_env_aux, 0) #Most conditions are non-negative
    
    # Physiological Emax (Env + clothing + wettedness) 
    Emax_wettedness_aux = Emax_wettedness(wmax, Psk_s, Pv_kPa, Re_cl, he_cof_aux, Icl, AD)  # [W] Maximum evaporative heat loss constrained by wettedness.
    Emax_wettedness_aux = np.maximum(Emax_wettedness_aux, 0) #Most conditions are non-negative     
    # =============================================================================
    # Estimation of Sweat requirements and Emax constrained to the possible sweat rate
    # =============================================================================
    #Skin wettedness and sweating efficiency
    wreq_surv = wreq_HSI_skin_wettedness(Ereq_aux, Emax_env_aux)  # [dimensionless] Required skin wettedness.
    r = Sweating_efficiency_r(wreq_surv)  # [dimensionless] Sweating efficiency.
    SRreq_aux = SRreq(Ereq_aux, r, Lh_vap)  # [L·h⁻¹] Required sweat rate for heat balance.

    SRmax = np.ones(tdb.shape)* srmax_abs  # [L·h⁻¹] Maximum sweat rate from profile.
    Emax_sweat_aux = Emax_sweat_rate(SRmax, Lh_vap, 1, r)  # [W] Maximum evaporative heat loss constrained by maximum sweat rate.

    # =============================================================================
    # Livability assessment
    # =============================================================================
   
    Mmax_W, Mmax_MET = livability_Mmax(Ereq_aux, Emax_wettedness_aux, Emax_sweat_aux, M, mass) 
    # =============================================================================
    # Survivability assessment
    # =============================================================================
    survivability_6H, flag_survivability_6H = Survivability(6,Ereq_aux,Emax_wettedness_aux,Emax_sweat_aux,SRreq_aux,SRmax,mass)
    survivability_3H, flag_survivability_3H = Survivability(3,Ereq_aux,Emax_wettedness_aux,Emax_sweat_aux,SRreq_aux,SRmax,mass)
    # =============================================================================
    # Identification of non-livable exposures
    # =============================================================================
    UHS_1p5MET = non_livable(Mmax_MET)

    
    return HEATLim(
        survive6H=survivability_6H,
        survive3H=survivability_3H,
        UHS_1p5MET=UHS_1p5MET,
        Mmax=Mmax_MET,
        Mmax_W=Mmax_W)
