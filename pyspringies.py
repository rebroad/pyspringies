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

    def add_spring(self, id, m1_id, m2_id, ks, kd, restlen):
        m1_index = np.where(self.masses['id'] == m1_id)[0]
        m2_index = np.where(self.masses['id'] == m2_id)[0]

        if len(m1_index) == 0 or len(m2_index) == 0:
            print(f"Warning: Cannot add spring {id}. Mass not found.")
            return

        if ks == 0:
            ks = self.default_ks
        if kd == 0:
            kd = self.default_kd
        new_spring = np.array([(id, m1_id, m2_id, ks, kd, restlen)], dtype=self.springs.dtype)
        self.springs = np.concatenate((self.springs, new_spring))

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
            m1_index = np.where(self.masses['id'] == spring['mass1'])[0]
            m2_index = np.where(self.masses['id'] == spring['mass2'])[0]

            if len(m1_index) > 0 and len(m2_index) > 0:
                m1, m2 = m1_index[0], m2_index[0]
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
        mask = ~self.masses['fixed']

        # Debug: Print max force
        max_force = np.max(np.sqrt(np.sum(forces**2, axis=1)))
        print(f"Max force: {max_force}")

        self.masses['vx'] += forces[:, 0] / self.masses['mass'] * self.dt
        self.masses['vy'] += forces[:, 1] / self.masses['mass'] * self.dt

        # Debug: Print max velocity
        max_velocity = np.max(np.sqrt(self.masses['vx']**2 + self.masses['vy']**2))
        print(f"Max velocity: {max_velocity}")

        self.masses['x'] += self.masses['vx'] * self.dt
        self.masses['y'] += self.masses['vy'] * self.dt
        self.apply_boundaries()

    def apply_boundaries(self):
        mask = ~self.masses['fixed']

        # Bottom wall
        if self.walls[3]:
            bottom_mask = self.masses['y'] < self.masses['radius'] & mask
        self.masses['y'][bottom_mask] = self.masses['radius'][bottom_mask]
        self.masses['vy'][bottom_mask] *= -self.masses['elastic'][bottom_mask]

        # Top wall
        if self.walls[0]:
            top_mask = self.masses['y'] > self.height - self.masses['radius'] & mask
            self.masses['y'][top_mask] = self.height - self.masses['radius'][top_mask]
            self.masses['vy'][top_mask] *= -self.masses['elastic'][top_mask]

        # Left wall
        if self.walls[1]:
            left_mask = self.masses['x'] < self.masses['radius'] & mask
            self.masses['x'][left_mask] = self.masses['radius'][left_mask]
            self.masses['vx'][left_mask] *= -self.masses['elastic'][left_mask]

        # Right wall
        if self.walls[2]:
            right_mask = self.masses['x'] > self.width - self.masses['radius'] & mask
            self.masses['x'][right_mask] = self.width - self.masses['radius'][right_mask]
            self.masses['vx'][right_mask] *= -self.masses['elastic'][right_mask]


    def calculate_derivative(self, mass, dx, dy):
        original_x, original_y = mass.x, mass.y
        mass.x += dx
        mass.y += dy
        fx, fy = self.calculate_forces(mass)
        mass.x, mass.y = original_x, original_y
        return mass.vx, mass.vy, fx / mass.mass, fy / mass.mass

def mass_radius(mass):
    return max(1, min(64, int(2 * np.log(4.0 * mass + 1.0))))

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
                        fixed = mass < 0
                        mass = abs(mass)
                        id = int(id)
                        radius = mass_radius(mass)
                        new_mass = np.array([(id, x, y, vx, vy, mass, elastic, radius, fixed)],
                                            dtype=space.masses.dtype)
                        space.masses = np.concatenate((space.masses, new_mass))
                    except ValueError as e:
                        print(f"Error parsing mass on line {line_number}: {e}")
                elif parts[0] == 'spng':
                    try:
                        id, m1, m2, ks, kd, restlen = map(float, parts[1:])
                        m1, m2 = int(m1), int(m2)
                        id = int(id)
                        new_spring = np.array([(id, m1, m2, ks, kd, restlen)],
                                              dtype=space.springs.dtype)
                        space.springs = np.concatenate((space.springs, new_spring))
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

    print(f"Loaded {len(space.masses)} masses and {len(space.springs)} springs.")

    # Calculate the bounding box of all masses
    min_x = np.min(space.masses['x'])
    max_x = np.max(space.masses['x'])
    min_y = np.min(space.masses['y'])
    max_y = np.max(space.masses['y'])

    print(f"Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")

    # Calculate scale and offset to fit the simulation in the window
    scale = min(space.width / (max_x - min_x), space.height / (max_y - min_y)) * 0.9
    offset_x = (space.width - (max_x - min_x) * scale) / 2
    offset_y = (space.height - (max_y - min_y) * scale) / 2

    print(f"Scale: {scale}, Offset: ({offset_x}, {offset_y})")

    # Set up the display
    screen = pygame.display.set_mode((space.width, space.height))
    clock = pygame.time.Clock()

    # Pre-create surfaces for masses and springs
    mass_surface = pygame.Surface((space.width, space.height), pygame.SRCALPHA)
    spring_surface = pygame.Surface((space.width, space.height), pygame.SRCALPHA)

    running = True
    frame_count = 0
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
            m1_index = np.where(space.masses['id'] == spring['mass1'])[0]
            m2_index = np.where(space.masses['id'] == spring['mass2'])[0]
            if len(m1_index) > 0 and len(m2_index) > 0:
                m1, m2 = m1_index[0], m2_index[0]
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

        frame_count += 1
        if frame_count % 60 == 0:  # Print debug info every 60 frames
            print(f"Frame {frame_count}")
            print(f"Mass positions: Min ({np.min(space.masses['x'])}, {np.min(space.masses['y'])}), "
                  f"Max ({np.max(space.masses['x'])}, {np.max(space.masses['y'])})")


    pygame.quit()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PySpringies simulation.")
    parser.add_argument("xsp_file", help="Path to the XSP file")
    args = parser.parse_args()

    main(args.xsp_file)
