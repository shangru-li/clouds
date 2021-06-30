import taichi as ti
import numpy as np
import time
from math import sin, cos, tan, radians
from fluid import trilerp, trilerp_smooth, Fluid
from thermo import start_thermo, vertical_scale

ti.init(arch=ti.gpu, debug=True)

interactive_mode = 1

inf = 1e10
eps = 1e-10
camera = None
color_size = 1200
color_resolution = (color_size, color_size)
fluid_dh = 100 # meters
fluid_edge = 10000 # meters
fluid_real_shape = (fluid_edge, fluid_edge * vertical_scale, fluid_edge)
fluid_shape = np.array(fluid_real_shape) / fluid_dh
color = ti.Vector.field(3, dtype=ti.f32, shape=color_resolution)
camera_data = ti.Vector.field(3, dtype=ti.f32, shape=(4,))
volumetric_data = ti.field(dtype=ti.f32, shape=ti.Vector(fluid_shape.astype(int)))

ray_depth = 100
if interactive_mode:
    ray_depth = 4
samples_per_pixel = 1000
if interactive_mode:
    samples_per_pixel = 50

def normalized(vec):
    return vec / (np.linalg.norm(vec) + eps)

light_angle = 0.0
light_y = 30.0
def rotate_light(factor):
    global light_angle
    rotate_speed = 4.0
    light_angle += factor * rotate_speed
    while light_angle >= 360: light_angle -= 360
    while light_angle < 0: light_angle += 360

class Camera:
    def __init__(self):
        self.length = 1.2
        self.theta = 20#0
        self.phi = 210#0
        self.rotate_speed = 5
        self.near = 0.001
        self.far = 10000
        self.fov = 90
        self.update_dependent_parameters()

    def update_dependent_parameters(self):
        y = sin(radians(self.theta))
        z = cos(radians(self.theta)) * cos(radians(self.phi))
        x = cos(radians(self.theta)) * sin(radians(self.phi))
        self.front = normalized([-x, -y, -z])
        self.right = normalized(np.cross(self.front, [0, 1, 0]))
        self.up = normalized(np.cross(self.right, self.front))
        self.position = np.array([x, y, z]) * self.length

    def rotate_phi(self, factor: float):
        self.phi += factor * self.rotate_speed
        while self.phi >= 360: self.phi -= 360
        while self.phi < 0: self.phi += 360
        self.update_dependent_parameters()

    def rotate_theta(self, factor: float):
        self.theta += factor * self.rotate_speed
        while self.theta >= 90: self.theta = 89
        while self.theta <= -90: self.theta = -89
        self.update_dependent_parameters()

@ti.kernel
def update_vector_data(vec: ti.template(), offset: ti.i32, x: ti.f32, y: ti.f32, z: ti.f32):
    vec[offset] = ti.Vector([x, y, z])

@ti.func
def ray_aabb_intersection(box_min: ti.template(), box_max: ti.template(), o: ti.template(), d: ti.template()):
    intersect = 1

    near_int = -inf
    far_int = inf

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    if near_int > far_int:
        intersect = 0
    return intersect, near_int, far_int

max_density = 0.1
sigma_t = 5.4#1.0
shadow_coef = 2.5#4.4#4.4#1.0  # 3.4 for orange
#alt_sigma_t = 3.5# 3.5

@ti.func
def ray_march_volume(box_min: ti.template(), box_max: ti.template(), o: ti.template(), d: ti.template(), t: ti.f32):
    dt = 0.03
    _t = dt
    _sum = 0.0
    while _t < t:
        _sum += dt * trilerp(volumetric_data, (o + _t * d - box_min) / (box_max - box_min) * volumetric_data.shape) * 15.0
        if _sum >= 1: 
            break
        _t += dt
    return _sum

@ti.func
def get_relative_density(box_min: ti.template(), box_max: ti.template(), o: ti.template()):
    return trilerp_smooth(volumetric_data, (o - box_min) / (box_max - box_min) * volumetric_data.shape) / max_density

@ti.func
def ray_march_volume_2(box_min: ti.template(), box_max: ti.template(), o: ti.template(), d: ti.template(), t: ti.f32, light_angle: ti.f32, sigma_t: ti.f32, shadow_coef: ti.f32):
    dt = 0.03
    _t = ti.random() * 3 * dt
    _sum = 0.0
    transmittance = 1.0
    rad_light_y = light_y * 3.14159 / 180
    rad_light_angle = light_angle * 3.14159 / 180
    light_dir = ti.Vector([ti.cos(rad_light_y) * ti.sin(rad_light_angle), ti.sin(rad_light_y), ti.cos(rad_light_y) * ti.cos(rad_light_angle)])
    light_color = 1.0
    while _t < t:
        curr_o = o + d * _t
        density = get_relative_density(box_min, box_max, curr_o)
        d_transmittance = ti.exp(-sigma_t * density * dt)
        transmittance *= d_transmittance
        intersect, near_int, far_int = ray_aabb_intersection(box_min, box_max, curr_o, light_dir)
        light_ray_transmittance = 1.0
        if intersect:
            __t = dt
            while __t < far_int:
                light_ray_density = get_relative_density(box_min, box_max, curr_o + __t * light_dir )
                light_ray_transmittance *= ti.exp(-sigma_t * shadow_coef * light_ray_density * dt)
                __t += dt
        else:
            print('point not inside')
        _sum += transmittance * light_ray_transmittance * light_color * (1.0 - d_transmittance) / sigma_t
        _t += dt
    return _sum, transmittance

@ti.func
def sample(box_min: ti.template(), box_max: ti.template(), o: ti.template(), d: ti.template(), sigma_t: ti.f32):
    t = 0.0
    ret = 1.0
    intersect, near_int, far_int = ray_aabb_intersection(box_min, box_max, o, d)
    has_hit = 0
    while True:
        t -= ti.log(1 - ti.random()) / sigma_t
        if t >= far_int:
            has_hit = 0
            break
        if (trilerp(volumetric_data, (o + t * d - box_min) / (box_max - box_min) * volumetric_data.shape) / max_density > ti.random()):
            ret = 0.9 #sigma_s / sigma_t
            has_hit = 1
            break
    return has_hit, o + d * t, ret

@ti.func
def tr(box_min: ti.template(), box_max: ti.template(), o: ti.template(), d: ti.template(), sigma_t: ti.f32, shadow_coef: ti.f32):
    t = 0.0
    ret = 1.0
    intersect, near_int, far_int = ray_aabb_intersection(box_min, box_max, o, d)
    cnt = 0
    while True:
        t -= ti.log(1 - ti.random()) / (sigma_t * shadow_coef)
        #if cnt > 10:
        #    print(t, far_int)
        if t >= far_int:
            break
        ret *= 1 - trilerp(volumetric_data, (o + t * d - box_min) / (box_max - box_min) * volumetric_data.shape) / max_density
        cnt += 1
    return ret

@ti.func
def coordinates(v1: ti.template()):
    v2 = ti.Vector([0.0, 0.0, 0.0])
    if v1.x > v1.y:
        v2 = ti.Vector([-v1.z, 0.0, v1.x]).normalized(eps)
    else:
        v2 = ti.Vector([0.0, v1.z, -v1.y]).normalized(eps)
    v3 = v1.cross(v2)
    return v2, v3

@ti.func
def spherical_direction(sin_theta, cos_theta, phi, x, y, z):
    return sin_theta * ti.cos(phi) * x + sin_theta * ti.sin(phi) * y + cos_theta * z

@ti.func
def phase(cos_theta, g):
    denom = 1 + g * g + 2 * g * cos_theta
    return (0.25 / 3.14159) * (1 - g * g) / (denom * ti.sqrt(denom))

g = -0.2

@ti.func
def sample_p(d: ti.template()):
    cos_theta = 0.0
    if g < 1e-3:
        cos_theta = 1 - 2 * ti.random()
    else:
        sqr_term = (1 - g * g) / (1 + g - 2 * g * ti.random())
        cos_theta = -(1 + g * g - sqr_term * sqr_term) / (2 * g)
    
    sin_theta = ti.sqrt(ti.max(0.0, 1 - cos_theta * cos_theta))
    phi = 2 * 3.14159 * ti.random()
    x, y = coordinates(d)
    wi = spherical_direction(sin_theta, cos_theta, phi, x, y, d)
    return wi

@ti.func
def delta_tracking(box_min: ti.template(), box_max: ti.template(), o: ti.template(), d: ti.template(), light_angle: ti.f32, sigma_t: ti.f32, shadow_coef: ti.f32):
    o = o + d * 1e-4
    beta = 1.0
    L = 0.0
    rad_light_y = light_y * 3.14159 / 180
    rad_light_angle = light_angle * 3.14159 / 180
    d_light = ti.Vector([ti.cos(rad_light_y) * ti.sin(rad_light_angle), ti.sin(rad_light_y), ti.cos(rad_light_y) * ti.cos(rad_light_angle)])
    light_intensity = 25.0
    _o = o
    _d = d
    leave_medium = 0
    for i in range(ray_depth):
        has_hit, _o, ret = sample(box_min, box_max, _o, _d, sigma_t)
        beta *= ret
        if has_hit:
            L += beta * tr(box_min, box_max, _o, d_light, sigma_t, shadow_coef) * light_intensity * phase(-_d.dot(d_light), g)
            _d = sample_p(_d)
        else:
            leave_medium = 1
            break
    return L, beta, leave_medium#tr(box_min, box_max, o, d, sigma_t, shadow_coef)

ray_march = 0

@ti.func
def gamma_correct(color):
    return ti.pow(color, 1 / 2.2)

@ti.func
def gamma_decorrect(color):
    return ti.pow(color, 2.2)

@ti.kernel
def render(light_angle: ti.f32, ray_march: ti.i32, sigma_t: ti.f32, shadow_coef: ti.f32):
    #background = ti.Vector([2.0, 45.0, 87.0]) / 255.0
    #background = ti.Vector([53.0, 81.0, 92.0]) / 255.0
    #background = ti.Vector([21.0, 34.0, 56.0]) / 255.0
    #background = ti.Vector([91.0, 33.0, 50.0]) / 255.0 # dark pink
    background = ti.Vector([50.0, 111.0, 166.0]) / 255.0 # sampled blue
    ###background = ti.Vector([255.0, 255.0, 255.0]) / 255.0 
    ##background = ti.Vector([246.0, 129.0, 2.0]) / 255.0 # orange

    background = gamma_decorrect(background)

    #base_color = ti.Vector([236.0, 138.0, 109.0]) / 255.0 # sample pink
    #base_color = ti.Vector([247.0, 202.0, 181.0]) / 255.0 # sample pink
    ##base_color = ti.Vector([242.0, 169.0, 134.0]) / 255.0 # sample pink
    #base_color = ti.Vector([255.0, 165.0, 0.0]) / 255.0 # orange
    #base_color = ti.Vector([255.0, 240.0, 150.0]) * 0.9 / 255.0 # sample pink
    base_color = ti.Vector([255.0, 181.0, 32.0]) * 0.9 / 255.0 # sample pink
    #base_color = ti.Vector([255.0, 255.0, 255.0]) / 255.0 # sample pink
    base_color = gamma_decorrect(base_color)
    zoom_size = 0.5
    front = camera_data[2]
    up = camera_data[1] * tan(camera.fov / 2) * zoom_size
    right = camera_data[0] * tan(camera.fov / 2) / color_resolution[0] * color_resolution[1] * zoom_size
    for u, v in color:
        ndc_u = float(u) / color_resolution[0] * 2 - 1
        ndc_v = float(v) / color_resolution[1] * 2 - 1
        position = camera_data[3] + right * ndc_u + up * ndc_v
        box_radius = 0.5
        box_min = ti.Vector([-1.0, -1.0 * vertical_scale, -1.0]) * box_radius
        box_max = ti.Vector([1.0, 1.0 * vertical_scale, 1.0]) * box_radius
        intersect, near_int, far_int = ray_aabb_intersection(box_min, box_max, position, front)
        if intersect:
            ret = 0.0
            background_coef = 0.0
            if ray_march == 1:
                background_coef = 1.0
                ret = ray_march_volume(box_min, box_max, position + near_int * front, front, far_int - near_int)
            elif ray_march == 2:
                ret, background_coef = ray_march_volume_2(box_min, box_max, position + near_int * front, front, far_int - near_int, light_angle, sigma_t, shadow_coef)
            else:
                iterations = samples_per_pixel
                for i in range(iterations):
                    L, transmittance, leave_medium = delta_tracking(box_min, box_max, position + near_int * front, front, light_angle, sigma_t, shadow_coef)
                    ret += L
                    if leave_medium:
                        background_coef += transmittance
                ret /= iterations
                background_coef /= iterations
            #color[u, v].fill(ret)
            color[u, v] = ret * base_color
            color[u, v] += background_coef * background

        else:
            color[u, v] = background
        color[u, v] = gamma_correct(color[u, v])

'''
relative_humidity = 0.269
Gamma_vapor_user = 1.0#0.295
E = 7.0
Gamma_heat = 3.5#0.3
perlin_coef = 0.09
'''

# -- cloud 1
'''
relative_humidity = 0.719
Gamma_vapor_user = 0.2
E = 12.2
Gamma_heat = 4.15#3.75
perlin_coef = 0.09
swirling_speed = 0.0
upward_speed = 0.0
'''

# -- cloud 2

relative_humidity = 0.4
Gamma_vapor_user = 0.9
E = 10.4
Gamma_heat = 5.4
perlin_coef = 0.037
swirling_speed = 0.0
upward_speed = 0.0


'''
relative_humidity = 0.75
Gamma_vapor_user = 0.2
E = 4.0#12.2
Gamma_heat = 5.0
perlin_coef = 0.09
swirling_speed = 0.0
upward_speed = 0.0s
'''

supercell = 0
# -- supercell 1
if supercell == 1:
    relative_humidity = 0.75
    Gamma_vapor_user = 0.25
    E = 0.8 #15.8
    Gamma_heat = 0.0
    perlin_coef = 0.09
    swirling_speed = 0.3#0.05  ##1
    upward_speed = 0.85#0.5  ##3

if supercell == 2:
    relative_humidity = 0.95
    Gamma_vapor_user = 0.25
    E = 15.8
    Gamma_heat = 0.0
    perlin_coef = 0.118
    swirling_speed = 1.0#0.05  ##1
    upward_speed = 3.0#0.5  ##3

if supercell == 3:
    relative_humidity = 1.0
    Gamma_vapor_user = 0.2
    E = 11.8
    Gamma_heat = 3.5
    perlin_coef = 0.09
    swirling_speed = 0.3
    upward_speed = 1.0
    


def update_control(gui: ti.template()):
    global relative_humidity, Gamma_vapor_user, E, Gamma_heat, perlin_coef, swirling_speed, upward_speed, sigma_t, shadow_coef, ray_march

    if gui.get_event((ti.GUI.PRESS, 'z')):
        ray_march = (ray_march + 1) % 3

    gui.get_event()
    if gui.is_pressed('a'): camera.rotate_phi(1)
    if gui.is_pressed('d'): camera.rotate_phi(-1)
    if gui.is_pressed('w'): camera.rotate_theta(-1)
    if gui.is_pressed('s'): camera.rotate_theta(1)
    if gui.is_pressed('q'): rotate_light(-1)
    if gui.is_pressed('e'): rotate_light(1)


    if gui.is_pressed('r'): 
        relative_humidity += 0.05
        if relative_humidity > 1.0: relative_humidity = 1.0
    if gui.is_pressed('f'): 
        relative_humidity -= 0.05
        if relative_humidity < 0.0: relative_humidity = 0.0

    if gui.is_pressed('t'):
        Gamma_vapor_user += 0.05
        if Gamma_vapor_user > 1.0: Gamma_vapor_user = 1.0
    if gui.is_pressed('g'):
        Gamma_vapor_user -= 0.05
        if Gamma_vapor_user < 0.0: Gamma_vapor_user = 0.0

    if gui.is_pressed('y'): 
        E += 0.2
        if E > 100.0: E = 100.0
    if gui.is_pressed('h'): 
        E -= 0.2
        if E < 0.0: E = 0.0
        
    if gui.is_pressed('u'):
        Gamma_heat += 0.05
        if Gamma_heat > 10.0: Gamma_heat = 10.0
    if gui.is_pressed('j'): 
        Gamma_heat -= 0.05
        if Gamma_heat < 0.0: Gamma_heat = 0.0
        
    if gui.is_pressed('c'):
        perlin_coef += 0.001
        if perlin_coef > 0.5: perlin_coef = 0.5
    if gui.is_pressed('x'): 
        perlin_coef -= 0.001
        if perlin_coef < 0.0: perlin_coef = 0.0
        
    if gui.is_pressed('i'):
        swirling_speed += 0.05
        if swirling_speed > 10.0: swirling_speed = 10.0
    if gui.is_pressed('k'): 
        swirling_speed -= 0.05
        if swirling_speed < 0.0: swirling_speed = 0.0

    if gui.is_pressed('o'):
        upward_speed += 0.05
        if upward_speed > 10.0: upward_speed = 10.0
    if gui.is_pressed('l'): 
        upward_speed -= 0.05
        if upward_speed < 0.0: upward_speed = 0.0

    if gui.is_pressed('b'):
        sigma_t += 0.1
        if sigma_t > 10.0: sigma_t = 10.0
    if gui.is_pressed('v'): 
        sigma_t -= 0.1
        if sigma_t < 0.1: sigma_t = 0.1

    if gui.is_pressed('m'):
        shadow_coef += 0.1
        if shadow_coef > 10.0: shadow_coef = 10.0
    if gui.is_pressed('n'): 
        shadow_coef -= 0.1
        if shadow_coef < 0.1: shadow_coef = 0.1

def update_camera_data():
    update_vector_data(camera_data, 0, camera.right[0], camera.right[1], camera.right[2])
    update_vector_data(camera_data, 1, camera.up[0], camera.up[1], camera.up[2])
    update_vector_data(camera_data, 2, camera.front[0], camera.front[1], camera.front[2])
    update_vector_data(camera_data, 3, camera.position[0], camera.position[1], camera.position[2])

@ti.kernel
def init_volumetric_data():
    _range = (ti.Vector(volumetric_data.shape) * 0.3, ti.Vector(volumetric_data.shape) * 0.7)
    for i, j, k in ti.ndrange((_range[0].x, _range[1].x), (_range[0].y, _range[1].y), (_range[0].z, _range[1].z)):
        volumetric_data[i, j, k] = 1
    
@ti.kernel

def to_volumetric(qf: ti.template(), scale: ti.f32):
    for i, j, k in volumetric_data:
        if j > 5:
            volumetric_data[i, j, k] = qf[i, j, k] * scale
        #if i == j == k == 0:
        #    print(qf[i, j, k])

gui = ti.GUI('Volume Render', color_resolution)

camera = Camera()
f = Fluid(fluid_real_shape, fluid_dh)
#init_volumetric_data()
#f.start(f.velocity, f.dye)

start_thermo(f.theta, f.q_v, f.dh, relative_humidity, Gamma_vapor_user, E, Gamma_heat, perlin_coef)

dt = 0
max_q_vs = 0.01
cnt = 0


result_dir = "./results"
video_manager = None
record_video = not interactive_mode
start_rec = 200
end_frame = start_rec + 240
if interactive_mode:
    start_rec = 0
    end_frame = 10e9
if record_video:
    video_manager = ti.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)
render_text = interactive_mode

while gui.running and cnt < end_frame:
    begin = time.time()
    update_control(gui)
    if not interactive_mode:
        rotate_light(1)
    update_camera_data()
    f.step(relative_humidity, Gamma_vapor_user, E, Gamma_heat, perlin_coef, swirling_speed, upward_speed)
    to_volumetric(f.q_c, 1 / max_q_vs)
    #volumetric_data = f.dye
    if cnt >= start_rec:
        render(light_angle, ray_march, sigma_t, shadow_coef)
    gui.set_image(color)
    if render_text:
        gui.text(content=f'relative humidity r+/f-  {relative_humidity:.4f}', pos=(0, 0.95), color=0xFFFFFF)
        gui.text(content=f'Gamma vapor t+/g-  {Gamma_vapor_user:.4f}', pos=(0, 0.90), color=0xFFFFFF)
        gui.text(content=f'E y+/h-  {E:.4f}', pos=(0, 0.85), color=0xFFFFFF)
        gui.text(content=f'Gamma_heat u+/j-  {Gamma_heat:.4f}', pos=(0, 0.80), color=0xFFFFFF)
        gui.text(content=f'perlin_coef c+/x-  {perlin_coef:.4f}', pos=(0, 0.75), color=0xFFFFFF)
        gui.text(content=f'swirling_speed i+/k-  {swirling_speed:.4f}', pos=(0, 0.7), color=0xFFFFFF)
        gui.text(content=f'upward_speed o+/l-  {upward_speed:.4f}', pos=(0, 0.65), color=0xFFFFFF)
        gui.text(content=f'render mode: z  {ray_march}', pos=(0, 0.6), color=0xFFFFFF)
        gui.text(content=f'sigma_t: b+/v-  {sigma_t:.4f}', pos=(0, 0.55), color=0xFFFFFF)
        gui.text(content=f'shadow_coef: m+/n-  {shadow_coef:.4f}', pos=(0, 0.5), color=0xFFFFFF)
    gui.show()
    
    if record_video and cnt >= start_rec:
        pixels_img = color.to_numpy()
        video_manager.write_frame(pixels_img)
        print(f'\rFrame {cnt}/{end_frame} is recorded', end='')
    
    end = time.time()
    dt = end - begin
    print(cnt)
    cnt += 1
    
if record_video:
    video_manager.make_video(gif=False, mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')