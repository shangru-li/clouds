import taichi as ti
import numpy as np
from taichi.lang.ops import rescale_index
from thermo import *
from noise import perlin

@ti.func
def sample(qf, u, v, w): # as ints
    I = ti.Vector([int(u), int(v), int(w)])
    I = max(0, min(qf.shape - ti.Vector([1, 1, 1]), I))
    return qf[I]

@ti.func
def sample_periodic(qf, u, v, w): # as ints
    I = ti.Vector([int(u), int(v), int(w)])
    while I.x >= qf.shape[0]: I.x -= qf.shape[0]
    while I.x < 0: I[0] += qf.shape[0]
    while I.z >= qf.shape[2]: I.z -= qf.shape[2]
    while I.z < 0: I[2] += qf.shape[2]
    I = max(0, min(qf.shape - ti.Vector([1, 1, 1]), I))
    return qf[I]

@ti.func
def lerp(vl, vr, frac): # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.func
def lerp_smooth(vl, vr, frac): # frac: [0.0, 1.0]
    frac = (1 - ti.cos(frac * 3.14159)) / 2.0
    return vl + frac * (vr - vl)

@ti.func
def trilerp(vf, p):
    u, v, w = p # p in float space
    r, s, t = u - 0.5, v - 0.5, w - 0.5 # in int space
    iu, iv, iw = ti.floor(r), ti.floor(s), ti.floor(t) # floor
    fu, fv, fw = r - iu, s - iv, t - iw # fract
    # YXZ
    c000 = sample(vf, iu, iv, iw)
    c001 = sample(vf, iu, iv, iw + 1)
    c010 = sample(vf, iu + 1, iv, iw)
    c011 = sample(vf, iu + 1, iv, iw + 1)
    c100 = sample(vf, iu, iv + 1, iw)
    c101 = sample(vf, iu, iv + 1, iw + 1)
    c110 = sample(vf, iu + 1, iv + 1, iw)
    c111 = sample(vf, iu + 1, iv + 1, iw + 1)
    c00 = lerp(c000, c001, fw)
    c01 = lerp(c010, c011, fw)
    c10 = lerp(c100, c101, fw)
    c11 = lerp(c110, c111, fw)
    c0 = lerp(c00, c01, fu)
    c1 = lerp(c10, c11, fu)
    return lerp(c0, c1, fv)
    
@ti.func
def trilerp_smooth(vf, p):
    u, v, w = p # p in float space
    r, s, t = u - 0.5, v - 0.5, w - 0.5 # in int space
    iu, iv, iw = ti.floor(r), ti.floor(s), ti.floor(t) # floor
    fu, fv, fw = r - iu, s - iv, t - iw # fract
    # YXZ
    c000 = sample(vf, iu, iv, iw)
    c001 = sample(vf, iu, iv, iw + 1)
    c010 = sample(vf, iu + 1, iv, iw)
    c011 = sample(vf, iu + 1, iv, iw + 1)
    c100 = sample(vf, iu, iv + 1, iw)
    c101 = sample(vf, iu, iv + 1, iw + 1)
    c110 = sample(vf, iu + 1, iv + 1, iw)
    c111 = sample(vf, iu + 1, iv + 1, iw + 1)
    c00 = lerp_smooth(c000, c001, fw)
    c01 = lerp_smooth(c010, c011, fw)
    c10 = lerp_smooth(c100, c101, fw)
    c11 = lerp_smooth(c110, c111, fw)
    c0 = lerp_smooth(c00, c01, fu)
    c1 = lerp_smooth(c10, c11, fu)
    return lerp_smooth(c0, c1, fv)

@ti.func
def trilerp_periodic(vf, p):
    u, v, w = p # p in float space
    r, s, t = u - 0.5, v - 0.5, w - 0.5 # in int space
    iu, iv, iw = ti.floor(r), ti.floor(s), ti.floor(t) # floor
    fu, fv, fw = r - iu, s - iv, t - iw # fract
    # YXZ
    c000 = sample_periodic(vf, iu, iv, iw)
    c001 = sample_periodic(vf, iu, iv, iw + 1)
    c010 = sample_periodic(vf, iu + 1, iv, iw)
    c011 = sample_periodic(vf, iu + 1, iv, iw + 1)
    c100 = sample_periodic(vf, iu, iv + 1, iw)
    c101 = sample_periodic(vf, iu, iv + 1, iw + 1)
    c110 = sample_periodic(vf, iu + 1, iv + 1, iw)
    c111 = sample_periodic(vf, iu + 1, iv + 1, iw + 1)
    c00 = lerp(c000, c001, fw)
    c01 = lerp(c010, c011, fw)
    c10 = lerp(c100, c101, fw)
    c11 = lerp(c110, c111, fw)
    c0 = lerp(c00, c01, fu)
    c1 = lerp(c10, c11, fu)
    return lerp(c0, c1, fv)

@ti.func
def backtrace_rk3(vf: ti.template(), p: ti.template(), dt: ti.f32, dh: ti.f32):
    v1 = trilerp(vf, p)
    p1 = p - 0.5 * dt * v1 / dh
    v2 = trilerp(vf, p1)
    p2 = p - 0.75 * dt * v2 / dh
    v3 = trilerp(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3) / dh
    return p

@ti.func
def backtrace_rk3_periodic(vf: ti.template(), p: ti.template(), dt: ti.f32, dh: ti.f32):
    v1 = trilerp_periodic(vf, p)
    p1 = p - 0.5 * dt * v1 / dh
    v2 = trilerp_periodic(vf, p1)
    p2 = p - 0.75 * dt * v2 / dh
    v3 = trilerp_periodic(vf, p2)
    p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3) / dh
    return p

@ti.data_oriented
class Fluid:

    def __init__(self, real_shape, dh):
        # params
        self.real_shape = real_shape
        self.shape = (np.array(real_shape) / dh).astype(int)
        self.dh = dh

        self.speed_scale = 300
        self.dt = self.speed_scale / 30

        self.p_jacobi_iters = 40
        self.curl_strength = 10.0
        self.density = 1.225

        self.step_cnt = 0

        # fields

        self.dye = ti.field(float, shape=self.shape)
        self.new_dye = ti.field(float, shape=self.shape)

        self.velocity = ti.Vector.field(3, float, shape=self.shape)
        self.new_velocity = ti.Vector.field(3, float, shape=self.shape)
        self.rain_velocity = ti.Vector.field(3, float, shape=self.shape)
        
        self.pressure = ti.field(float, shape=self.shape)
        self.new_pressure = ti.field(float, shape=self.shape)

        self.theta = ti.field(float, shape=self.shape)
        self.new_theta = ti.field(float, shape=self.shape)

        self.q_c = ti.field(float, shape=self.shape)
        self.new_q_c = ti.field(float, shape=self.shape)

        self.q_v = ti.field(float, shape=self.shape)
        self.new_q_v = ti.field(float, shape=self.shape)

        self.q_r = ti.field(float, shape=self.shape)
        self.new_q_r = ti.field(float, shape=self.shape)

        self.velocity_divergence = ti.field(float, shape=self.shape)
        self.velocity_curl = ti.Vector.field(3, float, shape=self.shape)

    ### main funcs

    @ti.kernel
    def advect(self, velocity: ti.template(), qf: ti.template(), new_qf: ti.template()):
        for i, j, k in velocity:
            p = ti.Vector([i, j, k]) + 0.5 # float space
            p = backtrace_rk3(velocity, p, self.dt, self.dh)
            new_qf[i, j, k] = trilerp(qf, p)
        self.swap(qf, new_qf)

    @ti.kernel
    def advect_periodic(self, velocity: ti.template(), qf: ti.template(), new_qf: ti.template()):
        for i, j, k in velocity:
            p = ti.Vector([i, j, k]) + 0.5 # float space
            p = backtrace_rk3_periodic(velocity, p, self.dt, self.dh)
            new_qf[i, j, k] = trilerp_periodic(qf, p)
        self.swap(qf, new_qf)


    @ti.kernel
    def vorticity(self, velocity: ti.template(), velocity_curl: ti.template()):
        for i, j, k in velocity:
            zy = sample(velocity, i, j + 1, k).z - sample(velocity, i, j - 1, k).z
            yz = sample(velocity, i, j, k + 1).y - sample(velocity, i, j, k - 1).y
            xz = sample(velocity, i, j, k + 1).x - sample(velocity, i, j, k - 1).x
            zx = sample(velocity, i + 1, j, k).z - sample(velocity, i - 1, j, k).z
            yx = sample(velocity, i + 1, j, k).y - sample(velocity, i - 1, j, k).y
            xy = sample(velocity, i, j + 1, k).x - sample(velocity, i, j - 1, k).x
            velocity_curl[i, j, k] = ti.Vector([zy - yz, xz - zx, yx - xy]) / (2 * self.dh)

    @ti.kernel
    def enhance_vorticity(self, velocity: ti.template(), velocity_curl: ti.template()):
        # anti-physics visual enhancement...
        for i, j, k in velocity:
            cl = sample(velocity_curl, i - 1, j, k)
            cr = sample(velocity_curl, i + 1, j, k)
            cb = sample(velocity_curl, i, j - 1, k)
            ct = sample(velocity_curl, i, j + 1, k)
            cff = sample(velocity_curl, i, j, k - 1)
            cbb = sample(velocity_curl, i, j, k + 1)
            cc = sample(velocity_curl, i, j, k)
            force = ti.Vector([cr.norm() - cl.norm(), ct.norm() - cb.norm(), cbb.norm() - cff.norm()]).normalized(1e-10).cross(cc) * self.curl_strength
            velocity[i, j, k] = min(max(velocity[i, j, k] + force * self.dt, -1e3), 1e3)

    @ti.kernel
    def divergence(self, velocity: ti.template(), velocity_divergence: ti.template()):
        for i, j, k in velocity:
            vl = sample(velocity, i - 1, j, k).x
            vr = sample(velocity, i + 1, j, k).x
            vb = sample(velocity, i, j - 1, k).y
            vt = sample(velocity, i, j + 1, k).y
            vff = sample(velocity, i, j, k - 1).z
            vbb = sample(velocity, i, j, k + 1).z
            if i == 0:
                vl = 0
            if i == self.shape[0] - 1:
                vr = 0
            if j == 0:
                vb = 0
            if j == self.shape[1] - 1:
                vt = 0
            if k == 0:
                vff = 0
            if k == self.shape[2] - 1:
                vbb = 0
            velocity_divergence[i, j, k] = (vr - vl + vt - vb + vbb - vff) / (2 * self.dh)
            

    @ti.kernel
    def pressure_jacobi_single(self, pressure: ti.template(), new_pressure: ti.template()):
        for i, j, k in pressure:
            pl = sample(pressure, i - 1, j, k)
            pr = sample(pressure, i + 1, j, k)
            pb = sample(pressure, i, j - 1, k)
            pt = sample(pressure, i, j + 1, k)
            pff = sample(pressure, i, j, k - 1)
            pbb = sample(pressure, i, j, k + 1)
            div = self.velocity_divergence[i, j, k]
            new_pressure[i, j, k] = (pl + pr + pb + pt + pff + pbb - self.density / self.dt * div * self.dh * self.dh) / 6

    @ti.kernel
    def subtract_gradient(self, velocity: ti.template(), pressure: ti.template()):
        for i, j, k in velocity:
            pl = sample(pressure, i - 1, j, k)
            pr = sample(pressure, i + 1, j, k)
            pb = sample(pressure, i, j - 1, k)
            pt = sample(pressure, i, j + 1, k)
            pff = sample(pressure, i, j, k - 1)
            pbb = sample(pressure, i, j, k + 1)
            velocity[i, j, k] -= self.dt / self.density * ti.Vector([pr - pl, pt - pb, pbb - pff]) / (2 * self.dh)

    @ti.func
    def swap(self, field: ti.template(), new_field: ti.template()):
        for i, j, k in field:
            field[i, j, k], new_field[i, j, k] = new_field[i, j, k], field[i, j, k]

    @ti.func
    def debug_field(self, qf: ti.template(), i: ti.i32, j: ti.i32, k: ti.i32, str1: ti.template()):
        print(str1, ti.Vector([i, j, k]), qf[i, j, k])

    @ti.kernel
    def debug_vel(self, i: ti.i32):
        print("=-=- v", i)
        self.debug_field(self.velocity, 25, 1, 25, 'vel')

    @ti.kernel
    def update_rain_velocity(self, velocity: ti.template(), rain_velocity: ti.template()):
        for i, j, k in velocity:
            rain_velocity[i, j, k] = velocity[i, j, k] + ti.Vector([0, -10.0, 0])
            #if j == 1:
            #    velocity[i, j, k] += ti.Vector([0, 100.0, 0])

    @ti.kernel
    def add_buoyancy(self, velocity: ti.template(), theta: ti.template(), q_v: ti.template(), perlin_coef: ti.f32):
        #print("---- B")
        for i, j, k in velocity:
            B = buoyancy(j * self.dh, theta[i, j, k], q_v[i, j, k])# * perlin(i, k, perlin_coef)

            # buoyancy hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee
            #if ti.abs(B) > 1e-3 and q_v[i, j, k] > 0: 
            velocity[i, j, k] += B * ti.Vector([0.0, 1.0, 0.0]) * self.dt

            '''
            # ====
            height = j * 100
            T_air = get_background_temperature(height)

            X_vapor = get_mole_fraction(q_v[i, j, k])
            M_thermal = get_thermal_molar_mass(X_vapor)

            pressure = get_background_pressure(height)
            Y_vapor = get_vapor_mass_fraction(X_vapor, M_thermal)
            Gamma_thermal = get_thermal_specific_heat_ratio(Y_vapor)
            T_thermal = get_thermal_temperature(theta[i, j, k], pressure, Gamma_thermal)

            Bb = get_buoyancy_force(M_thermal, T_air, T_thermal)

            q_vs = get_saturation_mixing_ratio(T_thermal, pressure)

            # ====
            for f in ti.static(range(10)):
                if i == k == 50 and (j == f * 10):
                    print(B, j, velocity[i,j,k], theta[i,j,k], q_v[i,j,k], '-', T_air, X_vapor, M_thermal, pressure, Gamma_thermal, T_thermal, [Bb, q_vs])
            if i == k == 50 and (j == 1 or j == 2 or j == 3):
                print(B, j, velocity[i,j,k], theta[i,j,k], q_v[i,j,k], '-', T_air, X_vapor, M_thermal, pressure, Gamma_thermal, T_thermal, [Bb, q_vs])
            '''

            

    @ti.kernel
    def add_force(self, velocity: ti.template(), q_v: ti.template()):
        for i, j, k in velocity:
            if j == 0:
                radius = 20
                #if ti.abs(i - 50) <= radius and ti.abs(k - 50) <= radius:
                scale = (-0.012 * j + 0.04) * perlin(i, k, 0.09)
                velocity[i, j, k] += ti.Vector([0, scale * q_v[i, j, k], 0]) * self.dt

    @ti.func
    def scale_wind_strength(self, relative_height):
        return -0.5 * 0.1 * (relative_height + 2) * (relative_height - 10)

    @ti.kernel
    def add_wind_force(self, velocity: ti.template(), swirling_speed: ti.f32, upward_speed: ti.f32):
        for i, j, k in velocity:
            relative_coords = ti.Vector([i, j, k]) / ti.Vector(self.shape) * 2.0 - 1.0
            z = relative_coords[2]
            x = relative_coords[0]
            norm = ti.Vector([z, x]).norm()
            norm_factor = 0.0
            if norm < 0.5:
                norm_factor = (1.0 + ti.cos(2 * norm * 3.14159)) / 2.0
            F = (ti.Vector([z, 0.0, -x]).normalized() * swirling_speed + ti.Vector([0.0, upward_speed, 0.0]))\
                 * norm_factor * self.scale_wind_strength(j / (10 * vertical_scale))
            F += ti.Vector([1.5,0,0])
            velocity[i, j, k] += F * self.dt
        return 

    def step(self, relative_humidity, Gamma_vapor_user, E, Gamma_heat, perlin_coef, swirling_speed, upward_speed):
        self.advect(self.velocity, self.velocity, self.new_velocity)
        self.update_rain_velocity(self.velocity, self.rain_velocity)
        
        if self.curl_strength:
            self.vorticity(self.velocity, self.velocity_curl)
            self.enhance_vorticity(self.velocity, self.velocity_curl)
        
        self.add_buoyancy(self.velocity, self.theta, self.q_v, perlin_coef)
        #self.add_force(self.velocity, self.q_v)
        self.add_wind_force(self.velocity, swirling_speed, upward_speed)
        
        self.divergence(self.velocity, self.velocity_divergence)
        for _ in range(self.p_jacobi_iters):
            self.pressure_jacobi_single(self.pressure, self.new_pressure)
            self.pressure, self.new_pressure = self.new_pressure, self.pressure
        self.subtract_gradient(self.velocity, self.pressure)
        
        self.advect(self.velocity, self.dye, self.new_dye)
        self.advect(self.velocity, self.theta, self.new_theta)
        self.advect_periodic(self.velocity, self.q_v, self.new_q_v)
        self.advect_periodic(self.velocity, self.q_c, self.new_q_c)
        self.advect(self.rain_velocity, self.q_r, self.new_q_r)

        step_thermodynamics(self.theta, self.q_v, self.q_c, self.q_r, self.dh)
        step_thermo_boundary(self.theta, self.q_v, self.dh, relative_humidity, Gamma_vapor_user, E, Gamma_heat, perlin_coef)
        self.step_cnt += 1