#!/bin/python

import pygame
import sys
import math
import argparse
import numpy as np
import cProfile
import pstats
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Mass:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float
    ay: float
    mass: float
    elastic: float
    radius: int
    fixed: bool = False

@dataclass
class Spring:
    id: int
    mass1: Mass
    mass2: Mass
    ks: float
    kd: float
    restlen: float

class Force:
    def __init__(self, enabled: bool = False, value: float = 0.0, misc: float = 0.0):
        self.enabled = enabled
        self.value = value
        self.misc = misc

class Space:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.masses = np.zeros(0, dtype=[('id', int), ('x', float), ('y', float), ('vx', float), ('vy', float), 
                                         ('mass', float), ('elastic', float), ('radius', int), ('fixed', bool)])
        self.springs = np.zeros(0, dtype=[('id', int), ('mass1', int), ('mass2', int), ('ks', float), ('kd', float), ('restlen', float)])
        self.dt = 0.01
        self.gravity = Force()
        self.center_mass = Force()
        self.pointer_attraction = Force()
        self.wall = Force()
        self.viscosity = 0.0
        self.stickiness = 0.0
        self.center_x = width / 2
        self.center_y = height / 2
        self.precision = 1.0
        self.adaptive_step = False
        self.grid_snap = 20.0
        self.grid_snap_enabled = False
        self.default_mass = 1.0
        self.default_elasticity = 1.0
        self.default_ks = 1.0
        self.default_kd = 0.1
        self.fix_mass = False
        self.show_springs = True
        self.center_id = -1
        self.walls = [False, False, False, False]  # top, left, right, bottom
        self.max_force = 100  # REB: Attempt to stop the explosions
        self.max_velocity = 100  # REB: Attempt to stop the explosions

    def add_mass(self, id, x, y, vx, vy, mass, elastic):
        fixed = mass < 0
        mass = abs(mass)
        if mass == 0:
            mass = self.default_mass
        if elastic == 0:
            elastic = self.default_elasticity
        radius = max(1, min(64, int(2 * np.log(4.0 * mass + 1.0))))
        new_mass = np.array([(id, x, y, vx, vy, mass, elastic, radius, fixed)], dtype=self.masses.dtype)
        self.masses = np.vstack((self.masses, new_mass))

    def add_spring(self, id, m1, m2, ks, kd, restlen):
        if ks == 0:
            ks = self.default_ks
        if kd == 0:
            kd = self.default_kd
        new_spring = np.array([(id, m1, m2, ks, kd, restlen)], dtype=self.springs.dtype)
        self.springs = np.vstack((self.springs, new_spring))

    def calculate_forces(self):
        forces = np.zeros((len(self.masses), 2))

        # Gravity
        if self.gravity.enabled:
            forces[:, 1] += self.masses['mass'] * self.gravity.value

        # Center of mass
        if self.center_mass.enabled:
            dx = self.center_x - self.masses['x']
            dy = self.center_y - self.masses['y']
            dist = np.sqrt(dx**2 + dy**2)
            mask = dist > 0
            force = np.zeros_like(dist)
            force[mask] = self.center_mass.value / (dist[mask] ** self.center_mass.misc)
            forces[:, 0] += force * dx * self.masses['mass']
            forces[:, 1] += force * dy * self.masses['mass']

        # Viscosity
        forces[:, 0] -= self.viscosity * self.masses['vx'] * self.masses['mass']
        forces[:, 1] -= self.viscosity * self.masses['vy'] * self.masses['mass']

        # Spring forces
        for spring in self.springs:
            m1, m2 = spring['mass1'], spring['mass2']
            if 0 <= m1 < len(self.masses) and 0 <= m2 < len(self.masses):
                dx = self.masses['x'][m2] - self.masses['x'][m1]
                dy = self.masses['y'][m2] - self.masses['y'][m1]
                distance = np.sqrt(dx**2 + dy**2)
                if distance > 0:
                    force = spring['ks'] * (distance - spring['restlen'])
                    damp = spring['kd'] * ((self.masses['vx'][m2] - self.masses['vx'][m1])*dx +
                                           (self.masses['vy'][m2] - self.masses['vy'][m1])*dy) / distance
                    total_force = (force - damp) / distance
                    forces[m1] += total_force * np.array([dx, dy])
                    forces[m2] -= total_force * np.array([dx, dy])
            else:
                print(f"Warning: Invalid mass index in spring {spring['id']} (m1={m1}, m2={m2})")

        if self.pointer_attraction.enabled:
            pass  # TODO

        if self.wall.enabled:
            pass  # TODO

        return forces

    def update(self):
        forces = self.calculate_forces()
        self.masses['vx'] += forces[:, 0] / self.masses['mass'] * self.dt
        self.masses['vy'] += forces[:, 1] / self.masses['mass'] * self.dt
        self.masses['x'] += self.masses['vx'] * self.dt
        self.masses['y'] += self.masses['vy'] * self.dt
        self.apply_boundaries()

    def apply_boundaries(self):
        mask = self.masses['x'] < self.masses['radius']
        self.masses['x'][mask] = self.masses['radius'][mask]
        self.masses['vx'][mask] *= -self.masses['elastic'][mask]

        mask = self.masses['x'] > self.width - self.masses['radius']
        self.masses['x'][mask] = self.width - self.masses['radius'][mask]
        self.masses['vx'][mask] *= -self.masses['elastic'][mask]

        mask = self.masses['y'] < self.masses['radius']
        self.masses['y'][mask] = self.masses['radius'][mask]
        self.masses['vy'][mask] *= -self.masses['elastic'][mask]

        mask = self.masses['y'] > self.height - self.masses['radius']
        self.masses['y'][mask] = self.height - self.masses['radius'][mask]
        self.masses['vy'][mask] *= -self.masses['elastic'][mask]

    def calculate_derivative(self, mass, dx, dy):
        original_x, original_y = mass.x, mass.y
        mass.x += dx
        mass.y += dy
        fx, fy = self.calculate_forces(mass)
        mass.x, mass.y = original_x, original_y
        return mass.vx, mass.vy, fx / mass.mass, fy / mass.mass

def load_xsp(filename: str) -> Space:
    space = Space(800, 600)
    try:
        with open(filename, 'r') as file:
            for line_number, line in enumerate(file, 1):
                parts = line.strip().split()
                if not parts or parts[0].startswith('#'):
                    continue
                if parts[0] == 'mass':
                    try:
                        id, x, y, vx, vy, mass, elastic = map(float, parts[1:])
                        space.add_mass(int(id), x, y, vx, vy, mass, elastic)
                    except ValueError as e:
                        print(f"Error parsing mass on line {line_number}: {e}")
                elif parts[0] == 'spng':
                    try:
                        id, m1, m2, ks, kd, restlen = map(float, parts[1:])
                        m1, m2 = int(m1), int(m2)
                        if 0 <= m1 < len(space.masses) and 0 <= m2 < len(space.masses):
                            space.add_spring(int(id), m1, m2, ks, kd, restlen)
                        else:
                            print(f"Warning on line {line_number}: Invalid mass index in spring {int(id)} (m1={m1}, m2={m2})")
                    except ValueError as e:
                        print(f"Error parsing spring on line {line_number}: {e}")
                elif parts[0] == 'frce':
                    try:
                        force_type, enabled, value, misc = int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
                        if force_type == 0:  # Gravity
                            space.gravity.enabled = enabled != 0
                            space.gravity.value = value
                            space.gravity.misc = misc
                        elif force_type == 1:  # Center of Mass
                            space.center_mass.enabled = enabled != 0
                            space.center_mass.value = value
                            space.center_mass.misc = misc
                        elif force_type == 2:  # Pointer Attraction
                            space.pointer_attraction_enabled = enabled != 0
                            space.pointer_attraction.value = value
                            space.pointer_attraction.misc = misc
                        elif force_type == 3:  # Wall
                            space.wall.enabled = enabled != 0
                            space.wall.value = value
                            space.wall.misc = misc
                    except ValueError as e:
                        print(f"Error parsing force: {e}")
                elif parts[0] == 'cmas':
                    space.default_mass = float(parts[1])
                elif parts[0] == 'elas':
                    space.default_elasticity = float(parts[1])
                elif parts[0] == 'kspr':
                    space.default_ks = float(parts[1])
                elif parts[0] == 'kdmp':
                    space.default_kd = float(parts[1])
                elif parts[0] == 'fixm':
                    space.fix_mass = int(parts[1]) != 0
                elif parts[0] == 'shws':
                    space.show_springs = int(parts[1]) != 0
                elif parts[0] == 'cent':
                    space.center_id = int(parts[1])
                elif parts[0] == 'visc':
                    space.viscosity = float(parts[1])
                elif parts[0] == 'stck':
                    space.stickiness = float(parts[1])
                elif parts[0] == 'step':
                    space.dt = float(parts[1])
                elif parts[0] == 'prec':
                    space.precision = float(parts[1])
                elif parts[0] == 'adpt':
                    space.adaptive_step = int(parts[1]) != 0
                elif parts[0] == 'gsnp':
                    space.grid_snap = float(parts[1])
                    space.grid_snap_enabled = int(parts[2]) != 0
                elif parts[0] == 'wall':
                    space.walls = [int(x) != 0 for x in parts[1:5]]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    if len(space.masses) == 0:
        print("Warning: No masses loaded from the file.")
    if len(space.springs) == 0:
        print("Warning: No springs loaded from the file.")

    return space

def main(xsp_file: str):
    profiler = cProfile.Profile()
    profiler.enable()

    pygame.init()
    space = load_xsp(xsp_file)

    if space is None or len(space.masses) == 0:
        print("Error: Failed to load valid simulation data.")
        pygame.quit()
        return

    space.dt = min(0.015, space.dt)  # REB - reduce explosions
    space.gravity.value = min(2.0, space.gravity.value)  # REB - reduce explosions

    #for spring in space.springs:
    #    spring.ks *= 0.5  # REB - reduce explosions

    # Set up the display
    screen = pygame.display.set_mode((space.width, space.height))
    clock = pygame.time.Clock()

    # Pre-create surfaces for masses and springs
    mass_surface = pygame.Surface((space.width, space.height), pygame.SRCALPHA)
    spring_surface = pygame.Surface((space.width, space.height), pygame.SRCALPHA)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        space.update()

        screen.fill((0, 0, 0))

        # Clear surfaces
        mass_surface.fill((0, 0, 0, 0))
        spring_surface.fill((0, 0, 0, 0))

        # Draw springs
        for spring in space.springs:
            m1, m2 = spring['mass1'], spring['mass2']
            pygame.draw.line(spring_surface, (255, 255, 255),
                             (int(space.masses['x'][m1]), int(space.height - space.masses['y'][m1])),
                             (int(space.masses['x'][m2]), int(space.height - space.masses['y'][m2])))

        # Draw masses
        for mass in space.masses:
            pygame.draw.circle(mass_surface, (255, 0, 0),
                               (int(mass['x']), int(space.height - mass['y'])), mass['radius'])

        # Blit surfaces to screen
        screen.blit(spring_surface, (0, 0))
        screen.blit(mass_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PySpringies simulation.")
    parser.add_argument("xsp_file", help="Path to the XSP file")
    args = parser.parse_args()

    main(args.xsp_file)
