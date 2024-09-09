#!/bin/python

import pygame
import sys
import math
import argparse
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
        self.masses: List[Mass] = []
        self.springs: List[Spring] = []
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
        self.max_velocity = 100  # REB: Attempt to stop the explosions
        self.integration_method = "euler"  # Default to Euler

    def add_mass(self, id, x, y, vx, vy, mass, elastic):
        fixed = False
        if mass < 0:
            mass = abs(mass)
            fixed = True

        if mass == 0:
            mass = self.default_mass
        if elastic == 0:
            elastic = self.default_elasticity

        radius = max(1, min(64, int(2 * math.log(4.0 * mass + 1.0))))
        self.masses.append(Mass(id, x, y, vx, vy, 0, 0, mass, elastic, radius, fixed))

    def add_spring(self, id, m1, m2, ks, kd, restlen):
        mass1 = next((m for m in self.masses if m.id == m1), None)
        mass2 = next((m for m in self.masses if m.id == m2), None)
        self.springs.append(Spring(id, mass1, mass2, ks, kd, restlen))
        if mass1 and mass2:
            if ks == 0:
                ks = self.default_ks
            if kd == 0:
                kd = self.default_kd
            self.springs.append(Spring(id, mass1, mass2, ks, kd, restlen))
        else:
            print(f"Warning: Could not create spring {id}. Mass not found.")

    def rk4_step(self):
        for mass in self.masses:
            if not mass.fixed:
                # RK4 implementation
                k1 = self.calculate_derivative(mass, 0, 0)
                k2 = self.calculate_derivative(mass, k1[0]*self.dt/2, k1[1]*self.dt/2)
                k3 = self.calculate_derivative(mass, k2[0]*self.dt/2, k2[1]*self.dt/2)
                k4 = self.calculate_derivative(mass, k3[0]*self.dt, k3[1]*self.dt)

                # Update position and velocity
                mass.x += self.dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
                mass.y += self.dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
                mass.vx += self.dt * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
                mass.vy += self.dt * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6

                # Apply boundary conditions
                self.apply_boundaries(mass)

    def update(self):
        max_force = 0
        max_velocity = 0
        for mass in self.masses:
            if not mass.fixed:
                fx, fy = self.calculate_forces(mass)
                force = math.sqrt(fx*fx + fy*fy)
                velocity = math.sqrt(mass.vx*mass.vx + mass.vy*mass.vy)
                max_force = max(max_force, force)
                max_velocity = max(max_velocity, velocity)

        print(f"Max force: {max_force:.2f}, Max velocity: {max_velocity:.2f}, dt: {self.dt:.6f}")

        # Call the appropriate integration method
        if self.adaptive_step:
            self.adaptive_step_update()
        elif self.integration_method == "euler":
            self.euler_step()
        elif self.integration_method == "rk4":
            self.rk4_step()

    def euler_step(self):
        for mass in self.masses:
            if not mass.fixed:
                # Calculate forces
                fx, fy = self.calculate_forces(mass)

                # Update velocity
                mass.vx += fx / mass.mass * self.dt
                mass.vy += fy / mass.mass * self.dt

                # Update position
                mass.x += mass.vx * self.dt
                mass.y += mass.vy * self.dt

                # Apply boundary conditions
                self.apply_boundaries(mass)

    def calculate_forces(self, mass):
        fx, fy = 0, 0

        if self.gravity.enabled:
            fy += mass.mass * self.gravity.value

        if self.center_mass.enabled:
            dx = self.center_x - mass.x
            dy = self.center_y - mass.y
            dist = math.sqrt(dx*dx + dy*dy) # TODO - add something here to avoid divide by zero?
            if dist > 0:
                force = self.center_mass.value / (dist ** self.center_mass.misc)
                fx += force * dx * mass.mass
                fy += force * dy * mass.mass

        fx -= self.viscosity * mass.vx * mass.mass
        fy -= self.viscosity * mass.vy * mass.mass

        for spring in self.springs:
            if spring.mass1 == mass or spring.mass2 == mass:
                other = spring.mass2 if spring.mass1 == mass else spring.mass1
                dx = other.x - mass.x
                dy = other.y - mass.y
                distance = math.sqrt(dx*dx + dy*dy) # TODO - add something here too?
                if distance > 0:
                    force = spring.ks * (distance - spring.restlen)
                    damp = spring.kd * ((other.vx - mass.vx)*dx + (other.vy - mass.vy)*dy) / distance
                    total_force = (force - damp) / distance
                    fx += total_force * dx / distance
                    fy += total_force * dy / distance

        if self.pointer_attraction.enabled:
            pass  # TODO

        if self.wall.enabled:
            pass  # TODO

        return fx, fy

    def calculate_derivative(self, mass, dx, dy):
        original_x, original_y = mass.x, mass.y
        mass.x += dx
        mass.y += dy
        fx, fy = self.calculate_forces(mass)
        mass.x, mass.y = original_x, original_y
        return mass.vx, mass.vy, fx / mass.mass, fy / mass.mass

    def apply_boundaries(self, mass):
        if mass.x < mass.radius:
            mass.x = mass.radius
            mass.vx *= -mass.elastic
        elif mass.x > self.width - mass.radius:
            mass.x = self.width - mass.radius
            mass.vx *= -mass.elastic

        if mass.y < mass.radius:
            mass.y = mass.radius
            mass.vy *= -mass.elastic
        elif mass.y > self.height - mass.radius:
            mass.y = self.height - mass.radius
            mass.vy *= -mass.elastic

        mass.vx = max(-self.max_velocity, min(self.max_velocity, mass.vx))  # REB: No explosions
        mass.vy = max(-self.max_velocity, min(self.max_velocity, mass.vy))  # REB: No explosions

def load_xsp(filename: str) -> Space:
    space = Space(800, 600)
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts or parts[0].startswith('#'):
                    continue
                if parts[0] == 'mass':
                    try:
                        id, x, y, vx, vy, mass, elastic = map(float, parts[1:])
                        space.add_mass(int(id), x, y, vx, vy, mass, elastic)
                    except ValueError as e:
                        print(f"Error parsing mass: {e}")
                elif parts[0] == 'spng':
                    try:
                        id, m1, m2, ks, kd, restlen = map(float, parts[1:])
                        space.add_spring(int(id), int(m1), int(m2), ks, kd, restlen)
                    except ValueError as e:
                        print(f"Error parsing spring: {e}")
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
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    return space

def main(xsp_file: str, integration_method: str):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    space = load_xsp(xsp_file)
    space.integration_method = integration_method

    space.dt = min(0.01, space.dt)  # REB - reduce explosions
    space.gravity.value = min(2.0, space.gravity.value)  # REB - reduce explosions

    for spring in space.springs:
        spring.ks *= 0.5  # REB - reduce explosions

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        space.update()

        screen.fill((0, 0, 0))

        for spring in space.springs:
            pygame.draw.line(screen, (255, 255, 255),
                             (int(spring.mass1.x), int(spring.mass1.y)),
                             (int(spring.mass2.x), int(spring.mass2.y)))

        for mass in space.masses:
            pygame.draw.circle(screen, (255, 0, 0),
                               (int(mass.x), int(mass.y)), mass.radius)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PySpringies simulation.")
    parser.add_argument("xsp_file", help="Path to the XSP file")
    parser.add_argument("--method", choices=["euler", "rk4"], default="euler",
                        help="Integretion method (default: euler)")
    args = parser.parse_args()

    main(args.xsp_file, args.method)
