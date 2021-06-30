import taichi as ti
from noise import perlin
ref_pressure = 101325
ref_temperature = 288.16 # sea level
c_p = 1004.68506 # constant pressure specific heat
gravity = 9.80665
M_air = 0.02896968 # dry air molar mass
R0 = 8.314462618 # universal gas constant
#gamma = 0.00976 # temperature lapse rate
# https://en.wikipedia.org/wiki/Atmospheric_pressure

M_water = 0.01802
Gamma_vapor = 1.33
Gamma_air = 1.4
R_d = 287 # dry air gas constant
L = 2.5 # latent heat

lapse_rate = 0.0065 
vertical_scale = 1.0

@ti.func
def get_background_pressure(height): # p
    return ref_pressure * (1 - lapse_rate * height / ref_temperature) ** 5.2561

@ti.func
def get_background_temperature(height): # T
    return ref_temperature - lapse_rate * height + lapse_rate * 1.0 * max(height - 8000.0 * vertical_scale, 0.0)

@ti.func
def get_thermal_molar_mass(X_vapor): # M_thermal
    return X_vapor * M_water + (1.0 - X_vapor) * M_air

@ti.func
def get_vapor_mass_fraction(X_vapor, M_thermal): # Y_vapor
    return X_vapor * M_water / M_thermal

@ti.func
def get_mole_fraction(q_i): # X_i
    return q_i / (q_i + 1.0)

@ti.func
def get_thermal_specific_heat_ratio(Y_vapor): # Gamma_thermal
    return Y_vapor * Gamma_vapor + (1.0 - Y_vapor) * Gamma_air

@ti.func
def get_buoyancy_force(M_thermal, T_air, T_thermal): # B
    return gravity * ((M_air * T_thermal) / (M_thermal * T_air) - 1.0)

@ti.func
def get_saturation_mixing_ratio(temperature, pressure): # q_vs
    return 380.16 / pressure * ti.exp(17.67 * (temperature - 273.15) / (temperature - 273.15 + 243.5))

@ti.func
def get_heat_capacity(Gamma_thermal, M_thermal): # c_p_thermal
    return (Gamma_thermal * R0) / (M_thermal * (Gamma_thermal - 1.0))

@ti.func
def get_potential_temperature(temperature, pressure): # dry dir, initial
    return temperature * (ref_pressure / pressure) ** (R_d / c_p)

@ti.func
def get_absolute_temperature(theta, pressure):
    return theta * (pressure / ref_pressure) ** (R_d / c_p)

@ti.func
def get_thermal_temperature(theta, pressure, Gamma_thermal): # T_thermal for buoyancy
    return theta * (pressure / ref_pressure) ** ((Gamma_thermal - 1.0) / Gamma_thermal)

@ti.func
def buoyancy(height, theta, q_v):
    T_air = get_background_temperature(height)

    X_vapor = get_mole_fraction(q_v)
    M_thermal = get_thermal_molar_mass(X_vapor)

    pressure = get_background_pressure(height)
    Y_vapor = get_vapor_mass_fraction(X_vapor, M_thermal)
    Gamma_thermal = get_thermal_specific_heat_ratio(Y_vapor)
    T_thermal = get_thermal_temperature(theta, pressure, Gamma_thermal)

    B = get_buoyancy_force(M_thermal, T_air, T_thermal)
    return B

@ti.kernel
def step_thermodynamics(theta: ti.template(), q_v: ti.template(), q_c: ti.template(), q_r: ti.template(), dh: ti.f32):
    for i, j, k in theta:
        height = j * dh
        pressure = get_background_pressure(height)
        temperature = get_background_temperature(height)
        T_thermal = get_absolute_temperature(theta[i, j, k], pressure)
        # ====
        q_vs = get_saturation_mixing_ratio(T_thermal, pressure)
        evaporation = min(q_vs - q_v[i, j, k], q_c[i, j, k])

        q_v[i, j, k] += evaporation
        q_c[i, j, k] -= evaporation
        autoconversion = 1.0 * max(q_c[i, j, k] - 0.001, 0.0)
        
        q_c[i, j, k] -= autoconversion
        q_r[i, j, k] += autoconversion
        
        accretion = 0.1 * q_c[i, j, k] * q_r[i, j, k]
        q_c[i, j, k] -= accretion
        q_r[i, j, k] += accretion
        
        q_c[i, j, k] = max(q_c[i, j, k], 0)

        X_vapor = get_mole_fraction(q_v[i, j, k])
        M_thermal = get_thermal_molar_mass(X_vapor)
        Y_vapor = get_vapor_mass_fraction(X_vapor, M_thermal)
        Gamma_thermal = get_thermal_specific_heat_ratio(Y_vapor)
        c_p_thermal = get_heat_capacity(Gamma_thermal, M_thermal)
        theta[i, j, k] += L / c_p_thermal * get_mole_fraction(-evaporation)

@ti.func
def heat_emission(base_temperature, E, Gamma_heat, noise):
    return base_temperature + E * (Gamma_heat * (2.0 * noise - 1.0) + 1.0)

@ti.func
def vapor_emission(relative_humidity, temperature, pressure, Gamma_vapor, noise):
    q_vs = get_saturation_mixing_ratio(temperature, pressure)
    return relative_humidity * q_vs * (Gamma_vapor * (2.0 * noise - 1.0) + 1.0)


@ti.kernel
def step_thermo_boundary(theta: ti.template(), q_v: ti.template(), dh: ti.f32, relative_humidity: ti.f32, Gamma_vapor_user: ti.f32, E: ti.f32, Gamma_heat: ti.f32, perlin_coef: ti.f32):
    for i, j, k in theta:
        if j == 0:
            noise = perlin(i, k, perlin_coef)
            theta[i, j, k] = heat_emission(ref_temperature, E, Gamma_heat, noise)
            q_v[i, j, k] = vapor_emission(relative_humidity, ref_temperature, ref_pressure, Gamma_vapor_user, noise)

@ti.kernel
def start_thermo(theta: ti.template(), q_v: ti.template(), dh: ti.f32, relative_humidity: ti.f32, Gamma_vapor_user: ti.f32, E: ti.f32, Gamma_heat: ti.f32, perlin_coef: ti.f32):
    for i, j, k in theta:
        height = j * dh
        pressure = get_background_pressure(height)
        temperature = get_background_temperature(height)
        theta[i, j, k] = get_potential_temperature(temperature, pressure)
        if j == 0:
            noise = perlin(i, k, perlin_coef)
            theta[i, j, k] = heat_emission(ref_temperature, E, Gamma_heat, noise)
            q_v[i, j, k] = vapor_emission(relative_humidity, ref_temperature, ref_pressure, Gamma_vapor_user, noise)
            ##if i == k == 50 and j == 0:
            ##    print(perlin_coef, q_v[i, j, k], relative_humidity, temperature, pressure, Gamma_vapor_user, noise)
            #if i == 50:
            #    print(q_v[i, j, k], i, k)
            #if i == k == 50:
            #    print(theta[i, j, k], q_v[i, j, k], temperature, pressure, get_saturation_mixing_ratio(temperature, pressure))
            