
from typing import Tuple
import itertools
import numpy as np
from numpy.core.fromnumeric import clip

Vector2 = Tuple[float, float]
Vector3 = Tuple[float, float, float]

def area_of_intersecting_circles(rad_1, rad_2, center_dist):
    if center_dist <= max(rad_1, rad_2) - min(rad_1, rad_2):
        return np.pi*min(rad_1, rad_2)**2

    if center_dist == rad_1 + rad_2:
        return 0
    
    if center_dist > rad_1 + rad_2:
        raise ValueError("Cannot calculate the area of non-intersecting circles")
    
    part_1 = (rad_1**2)*np.arccos((center_dist**2 + rad_1**2 - rad_2**2)/(2*center_dist*rad_1))
    part_2 = (rad_2**2)*np.arccos((center_dist**2 + rad_2**2 - rad_1**2)/(2*center_dist*rad_2))
    part_3 = 0.5*np.sqrt((-center_dist + rad_1 + rad_2)*(center_dist + rad_1 - rad_2)*(center_dist - rad_1 + rad_2)*(center_dist + rad_1 + rad_1))
    return part_1 + part_2 - part_3

class ConeSensorModel:
    FOV_RAD = np.deg2rad(25)

    def __init__(self, backdrop_dist: float, apple_rad: float, stddev: float, rng: np.random.Generator):
        self.backdrop_dist = backdrop_dist
        self.apple_rad = apple_rad
        self.stddev = stddev
        self.rng = rng

    def calc_per_apple_in_fov(self, x, y, z) -> float:
        '''calculate the percantage of the FOV which contains apple'''
        fov_at_apple_rad = z*np.tan(ConeSensorModel.FOV_RAD/2)
        try:
            area_apple_in_fov = area_of_intersecting_circles(
                self.apple_rad, fov_at_apple_rad, np.sqrt(x**2 + y**2))
        except ValueError:
            return 0
        return clip(area_apple_in_fov/(np.pi * fov_at_apple_rad**2), 0, 1)

    def measure(self, apple_pos: Vector3, meas_pos: Tuple[Vector2, Vector2], est_apple_pos: Vector3) -> Tuple[Vector3, np.ndarray]:
        rx, ry, rz = apple_pos
        ex, ey, ez = est_apple_pos
        (mx, my), (varx, vary) = meas_pos

        # calculate the percantage of the FOV which contains apple
        per_apple_in_fov = self.calc_per_apple_in_fov(rx, ry, rz)

        # print(f'Apple fov: {per_apple_in_fov*100:.2f}%')
        # > 80% means the apple is basically correct
        if per_apple_in_fov >= .8:
            mz =  rz - self.apple_rad + self.rng.normal(0, self.stddev)

        # < 80% means we get a linear interpolation between the apples center
        # and the backdrop
        else:
            temp_z = rz + (self.backdrop_dist - rz)*(1 - per_apple_in_fov)
            mz = temp_z + self.rng.normal(0, self.stddev)
        
        # determine if the apple is likely to be >80% of the FOV (we'll use 1.5sigma)
        STD_RANGE = 1.5
        x_err = np.sqrt(varx)*STD_RANGE
        minmax_x = (mx + x_err, mx - x_err, mx)
        y_err = np.sqrt(vary)*STD_RANGE
        minmax_y = (my + y_err, my - y_err, my)
        z_err = self.stddev*1.5
        all_pos_per_apple_in_fov = [self.calc_per_apple_in_fov(x, y, mz + z_err) for x, y in itertools.product(minmax_x, minmax_y)]
        
        # compute predicted varience
        if min(all_pos_per_apple_in_fov) < .7:
            # this measurement is unlikely to be accurate, therefore our variance is maximum
            varz = self.backdrop_dist**2
            covzcx = 0
            covzcy = 0
        else:
            mes_per_apple_in_fov = self.calc_per_apple_in_fov(mx, my, ez)
            if mes_per_apple_in_fov >= .95:
                varz = self.stddev**2 + (self.apple_rad/2)**2
                covzcx = 0
                covzcy = 0
            else:
                expected_fov_at_apple_rad = ez*np.tan(ConeSensorModel.FOV_RAD/2)
                expected_per_apple_in_fov = self.calc_per_apple_in_fov(ex, ey, ez)

                d_var_est = varx + vary
                a_var_est = (1/4 + (1/(2*self.apple_rad))**2 + (1/(2*expected_fov_at_apple_rad))**2)*d_var_est
                varz = ((self.backdrop_dist - ez)*(1 - expected_per_apple_in_fov))**2 + max((self.backdrop_dist - mz)**2*a_var_est**2, 0)
                
                eacx = np.pi*ex - (1/2 + 1/(2*self.apple_rad) + 1/(2*expected_fov_at_apple_rad))*(varx + ex**2 + ey)
                covzcx = (self.backdrop_dist - ez)*eacx
                eacy = np.pi*ey - (1/2 + 1/(2*self.apple_rad) + 1/(2*expected_fov_at_apple_rad))*(vary + ex + ey**2)
                covzcy = (self.backdrop_dist - ez)*eacy
    
        var = np.array([
            [varx, 0, 0,],
            [0, vary, covzcy],
            [0, covzcx, varz]
        ])
        return (mx, my, mz), np.maximum(var, 0)


class NormalCameraModel:
    def __init__(self, camera_dev: float, rng: np.random.Generator):
        self.camera_dev = camera_dev
        self.rng = rng

    def measure(self, apple_pos: Vector3) -> Tuple[Vector2, Vector2]:
        rx, ry, _ = apple_pos
        x = rx + self.rng.normal(0, self.camera_dev)
        y = ry + self.rng.normal(0, self.camera_dev)
        return (x, y), (self.camera_dev**2, self.camera_dev**2)

class AppleModel:
    def __init__(self, initial_pos: Vector3, speed: float, delta_t_ms: int, camera: NormalCameraModel, sensor: ConeSensorModel):
        self.pos_real = initial_pos
        self.speed = speed
        self.delta_t_ms = delta_t_ms
        self.cur_time = 0
        self.camera = camera
        self.sensor = sensor
        self._last_velocity = np.array((0, 0, 0))

    def step(self, last_est_pos: Vector3) -> Tuple[Tuple[Vector3, np.ndarray], np.ndarray]:
        delta_t = self.delta_t_ms/1000
        if self.cur_time != 0:
            # move the robot towards the apple at a fixed speed
            new_velocity = (self.pos_real / np.linalg.norm(self.pos_real)) * -self.speed
            self.pos_real = self.pos_real + new_velocity*delta_t
        else:
            new_velocity = np.array((0, 0, 0))

        # compute control vector
        new_accel = (new_velocity - self._last_velocity)/delta_t
        control = np.transpose(np.array([new_velocity[0], new_accel[0], new_velocity[1], new_accel[1], new_velocity[2], new_accel[2]]))
        self._last_velocity = new_velocity

        # step time
        self.cur_time += self.delta_t_ms / 1000

        # take a measurement
        cam = self.camera.measure(self.pos_real)
        return self.sensor.measure(self.pos_real, cam, last_est_pos), control

if __name__ == '__main__':
    np.seterr(all='raise')
    rng = np.random.default_rng()
    model = AppleModel((0, 0, 1), 5, 10, NormalCameraModel(0.01, rng), ConeSensorModel(1.2, 0.08, 0.005, rng))

    while True:
        meas, var = model.step()
        print(meas)
        print(np.sqrt(var))
        print(model.pos_real)