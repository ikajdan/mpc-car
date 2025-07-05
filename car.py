import os

import numpy as np
import pygame

CAR_CONFIG = {
    "width_multiplier": 1,
    "wheel_offset_multiplier": 0.4,
    "wheel_width_multiplier": 0.2,
    "wheel_height_multiplier": 0.4,
    "wheel_color": (30, 30, 30),
}


class Car(pygame.sprite.Sprite):
    def __init__(self, L: int, car_image_path: str):
        super().__init__()

        if not pygame.get_init():
            pygame.init()
            pygame.display.set_mode((1, 1))

        self.L = L
        self.offset = pygame.Vector2((self.L // 2, 0))
        self.pos = pygame.Vector2((0, 0))
        self.theta = 0
        self.delta = 0

        self.wheel_offset = int(CAR_CONFIG["wheel_offset_multiplier"] * self.L)
        self.width = int(CAR_CONFIG["width_multiplier"] * self.L)
        self.height = int(self.wheel_offset * 2 + self.L)
        self.wheel_width = int(CAR_CONFIG["wheel_width_multiplier"] * self.L)
        self.wheel_height = int(CAR_CONFIG["wheel_height_multiplier"] * self.L)

        padding = int(self.wheel_height * 2)

        self.canvas_size = (self.width + 2 * padding, self.height + 2 * padding)

        self.base_image = pygame.Surface(self.canvas_size, pygame.SRCALPHA)
        self.base_image.fill((0, 0, 0, 0))

        self.car_body = pygame.image.load(car_image_path).convert_alpha()
        self.car_body = pygame.transform.scale(self.car_body, (self.width, self.height))

        self.wheel = pygame.Surface(
            (self.wheel_width, self.wheel_height), pygame.SRCALPHA
        )
        self.wheel.fill(CAR_CONFIG["wheel_color"])

        rear_wheel_spacing = int(self.width * 0.45)

        self.car_body.blit(
            self.wheel,
            self.wheel.get_rect(
                center=(
                    self.width // 2 - rear_wheel_spacing,
                    self.height - self.wheel_offset,
                )
            ),
        )

        self.car_body.blit(
            self.wheel,
            self.wheel.get_rect(
                center=(
                    self.width // 2 + rear_wheel_spacing,
                    self.height - self.wheel_offset,
                )
            ),
        )

        self.front_wheel = pygame.Surface(
            (int(self.wheel_width * 0.8), self.wheel_height), pygame.SRCALPHA
        )
        self.front_wheel.fill(CAR_CONFIG["wheel_color"])

        self.base_image.blit(self.car_body, (padding, padding))

        self.image = self.base_image.copy()
        self.rect = self.image.get_rect()

    def update(self, x: int, y: int, theta: float, delta: float):
        self.pos = pygame.Vector2((x, y))
        self.theta = theta
        self.delta = delta

        updated_image = self.base_image.copy()

        self._rotate_front_wheels(updated_image)
        self.image = updated_image

        self._rotate()

    def _rotate(self):
        self.image = pygame.transform.rotozoom(self.image, 90 - self.theta, 1)
        offset_rotated = self.offset.rotate(self.theta)
        self.rect = self.image.get_rect(center=self.pos + offset_rotated)

    def _rotate_front_wheels(self, updated_image):
        rotated_wheel = pygame.transform.rotate(self.front_wheel, self.delta)

        wheel_spacing = int(self.width * 0.45)
        padding = int(self.wheel_height * 2)

        updated_image.blit(
            rotated_wheel,
            rotated_wheel.get_rect(
                center=(
                    self.width // 2 - wheel_spacing + padding,
                    self.wheel_offset + padding,
                )
            ),
        )

        updated_image.blit(
            rotated_wheel,
            rotated_wheel.get_rect(
                center=(
                    self.width // 2 + wheel_spacing + padding,
                    self.wheel_offset + padding,
                )
            ),
        )


class Setpoint(pygame.sprite.Sprite):
    LINE_COLOR = (10, 10, 10)
    ARROW_LENGTH = 70
    ARROW_HEAD_SIZE = 15

    def __init__(self, x: int, y: int, theta: float):
        super().__init__()

        self.pos = pygame.Vector2((x, y))
        self.theta = theta

        buffer = 20
        self.image = pygame.Surface(
            (self.ARROW_LENGTH * 2 + buffer, self.ARROW_LENGTH * 2 + buffer),
            pygame.SRCALPHA,
        )
        self.original_image = self.image.copy()

        self._draw_angle_arrow()

        self.rect = self.image.get_rect(center=self.pos)

    def _draw_angle_arrow(self):
        self.original_image.fill((0, 0, 0, 0))

        center = (self.ARROW_LENGTH + 10, self.ARROW_LENGTH + 10)

        arrow_x = self.ARROW_LENGTH * np.cos(np.pi)
        arrow_y = self.ARROW_LENGTH * np.sin(np.pi)

        pygame.draw.line(
            self.original_image,
            self.LINE_COLOR,
            center,
            (center[0] + arrow_x, center[1] + arrow_y),
            4,
        )

        shift_forward = 5
        arrow_tip = (
            center[0] + arrow_x + shift_forward * np.cos(np.pi),
            center[1] + arrow_y + shift_forward * np.sin(np.pi),
        )

        left_wing = (
            arrow_tip[0] - self.ARROW_HEAD_SIZE * np.cos(np.pi / 4 + np.pi),
            arrow_tip[1] - self.ARROW_HEAD_SIZE * np.sin(np.pi / 4 + np.pi),
        )
        right_wing = (
            arrow_tip[0] - self.ARROW_HEAD_SIZE * np.cos(-np.pi / 4 + np.pi),
            arrow_tip[1] - self.ARROW_HEAD_SIZE * np.sin(-np.pi / 4 + np.pi),
        )

        pygame.draw.polygon(
            self.original_image, self.LINE_COLOR, [arrow_tip, left_wing, right_wing]
        )

    def update(self, x: int, y: int, theta: float):
        self.pos = pygame.Vector2((x, y))
        self.theta = theta

        rotated_image = pygame.transform.rotate(
            self.original_image, -np.degrees(self.theta)
        )
        self.image = rotated_image
        self.rect = self.image.get_rect(center=self.pos)


class CarEnv:
    ENVIROMENT_COLOR = (128, 128, 128)

    def __init__(
        self,
        L: int,
        car_image_path: str,
        setpoint: np.ndarray,
        env_size: tuple,
        t_step: float,
        notebook: bool = False,
    ):
        self.t_step = t_step
        self.notebook = notebook
        x, y, theta, _ = setpoint
        self.car = Car(L=L, car_image_path=car_image_path)
        self.setpoint = Setpoint(x=x, y=y, theta=theta)

        if self.notebook:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        else:
            os.environ["SDL_VIDEODRIVER"] = "x11"

        pygame.init()
        pygame.display.set_caption("MPC Car Simulation")
        self.screen = pygame.display.set_mode(env_size)
        self.clock = pygame.time.Clock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def make_step(self, x0: np.ndarray):
        try:
            x, y, theta, delta = x0
            self.screen.fill(self.ENVIROMENT_COLOR)
            self.car.update(x, y, np.rad2deg(theta), np.rad2deg(delta))
            self.screen.blit(self.car.image, self.car.rect)
            self.screen.blit(self.setpoint.image, self.setpoint.rect)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.setpoint.pos = pygame.Vector2((mouse_x, mouse_y))
                    self.setpoint.rect.center = self.setpoint.pos

                    x0[0] = mouse_x
                    x0[1] = mouse_y

            self.clock.tick(1 / self.t_step)

        except KeyboardInterrupt:
            self.close()

    def close(self):
        pygame.quit()
