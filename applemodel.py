
from typing import Tuple
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

    def measure(self, apple_pos: Vector3, meas_pos: Tuple[Vector2, Vector2], est_apple_pos: Vector3) -> Tuple[Vector3, np.ndarray]:
        rx, ry, rz = apple_pos
        ex, ey, ez = est_apple_pos
        (mx, my), (varx, vary) = meas_pos

        # calculate the percantage of the FOV which contains apple
        fov_at_apple_rad = rz*np.tan(ConeSensorModel.FOV_RAD/2)
        area_apple_in_fov = area_of_intersecting_circles(
            self.apple_rad, fov_at_apple_rad, np.sqrt(rx**2 + ry**2))
        per_apple_in_fov = clip(area_apple_in_fov/(np.pi * fov_at_apple_rad**2), 0, 1)

        # print(f'Apple fov: {per_apple_in_fov*100:.2f}%')
        # > 80% means the apple is basically correct
        if per_apple_in_fov >= .8:
            mz =  rz - self.apple_rad + self.rng.normal(0, self.stddev)

        # < 80% means we get a linear interpolation between the apples center
        # and the backdrop
        else:
            temp_z = rz + (self.backdrop_dist - rz)*(1 - per_apple_in_fov)
            mz = temp_z + self.rng.normal(0, self.stddev)
        
        # compute predicted varience
        expected_fov_at_apple_rad = ez*np.tan(ConeSensorModel.FOV_RAD/2)
        mes_area_apple_in_fov = area_of_intersecting_circles(
            self.apple_rad, expected_fov_at_apple_rad, np.sqrt(mx**2 + my**2))
        mes_per_apple_in_fov = clip(mes_area_apple_in_fov / (np.pi * expected_fov_at_apple_rad**2), 0, 1)
        # print(f'Measured apple fov: {mes_per_apple_in_fov*100:.2f}%')

        if mes_per_apple_in_fov >= .95:
            var = np.array([
                [varx, 0, 0],
                [0, vary, 0],
                [0, 0, self.stddev**2 + (self.apple_rad/2)**2]
            ])
        else:
            expected_area_apple_in_fov = area_of_intersecting_circles(
                self.apple_rad, expected_fov_at_apple_rad, np.sqrt(ex**2 + ey**2))
            expected_per_apple_in_fov = clip(expected_area_apple_in_fov/(np.pi*expected_fov_at_apple_rad**2), 0, 1)

            d_var_est = varx + vary
            a_var_est = (1/4 + (1/(2*self.apple_rad))**2 + (1/(2*expected_fov_at_apple_rad))**2)*d_var_est
            varz = ((self.backdrop_dist - ez)*(1 - expected_per_apple_in_fov))**2 + max((self.backdrop_dist - mz)**2*a_var_est**2, 0)
            
            eacx = np.pi*ex - (1/2 + 1/(2*self.apple_rad) + 1/(2*expected_fov_at_apple_rad))*(varx + ex**2 + ey)
            covzcx = max((self.backdrop_dist - ez)*eacx, 0)
            eacy = np.pi*ey - (1/2 + 1/(2*self.apple_rad) + 1/(2*expected_fov_at_apple_rad))*(vary + ex + ey**2)
            covzcy = max((self.backdrop_dist - ez)*eacy, 0)

            var = np.array([
                [varx,   0,      covzcx],
                [0,      vary,   covzcy],
                [covzcx, covzcy, varz]
            ])
        return (mx, my, mz), var


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

    def step(self) -> Tuple[Vector3, np.ndarray]:
        if self.cur_time != 0:
            # move the robot towards the apple at a fixed speed
            self.pos_real = self.pos_real + (self.pos_real / np.linalg.norm(self.pos_real)) * -self.speed * self.delta_t_ms / 1000
        self.cur_time += self.delta_t_ms / 1000

        # take a measurement
        cam = self.camera.measure(self.pos_real)
        return self.sensor.measure(self.pos_real, cam, self.pos_real)

    def get_control_vector(self) -> np.ndarray:
        if self.cur_time == 0:
            return np.transpose(np.array([0, 0, 0, 0, -self.speed, 0])) # todo: why /2?
        else:
            return np.transpose(np.array([0, 0, 0, 0, 0, 0]))

if __name__ == '__main__':
    np.seterr(all='raise')
    rng = np.random.default_rng()
    model = AppleModel((0, 0, 1), 5, 10, NormalCameraModel(0.01, rng), ConeSensorModel(1.2, 0.08, 0.005, rng))

    while True:
        meas, var = model.step()
        print(meas)
        print(np.sqrt(var))
        print(model.pos_real)