import taichi as ti


@ti.func
def fl(uv: ti.template()):
    return ti.Vector([ti.floor(float(uv.x)), ti.floor(float(uv.y))])

@ti.func
def random2(p: ti.template()):
    f = ti.sin(ti.Vector([p.dot(ti.Vector([127.1, 311.7])), p.dot(ti.Vector([269.5, 183.3]))])) * 43758.5453
    return f - fl(f)

# --- perlin
@ti.func
def surflet(uv: ti.template(), grid_point: ti.template()):
    t2 = ti.abs(uv - grid_point)
    t = ti.Vector([1, 1]) - 6 * ti.pow(t2, 5) + 15 * ti.pow(t2, 4) - 10 * pow(t2, 3)
    gradient = random2(grid_point) * 2 - ti.Vector([1, 1])
    diff = uv - grid_point
    height = diff.dot(gradient)
    return height * t.x * t.y


@ti.func
def perlin_noise(uv: ti.template()):
    surflet_sum = 0.0
    for i in range(2):
        for j in range(2):
            surflet_sum += surflet(uv, fl(uv) + ti.Vector([i, j]))
    return surflet_sum

@ti.func
def perlin(i, j, tiling):
    n = perlin_noise(ti.Vector([i, j]) * tiling) + 0.5
    if n > 1.0:
        n = 1.0
    if n < 0.1:
        n = 0.1
    return n #0.5 * (perlin_noise(ti.Vector([i, j]) * tiling) + 1) # 0.03
    

@ti.kernel
def noisy(color: ti.template()):
    scale = 1
    for i, j in color:
        #x = i / color_resolution[0] * scale
        #y = j / color_resolution[1] * scale
        color[i, j] = perlin(i, j, 0.09)#fbm(x, y)
        if i == 50: print(color[i,j])

if __name__ == '__main__':
    ti.init(arch=ti.gpu, debug=True)
    color_size = 100
    color_resolution = (color_size, color_size)
    gui = ti.GUI('Noise', color_resolution)
    color = ti.field(dtype=ti.f32, shape=color_resolution)
    noisy(color)
    while gui.running:
        gui.set_image(color)
        gui.show()
