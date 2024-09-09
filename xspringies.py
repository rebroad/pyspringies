#!/bin/python

import pygame
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

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
    def __init__(self, enabledL bool = False, value: float = 0.0, misc: float = 0.0):
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
        self.center_x = width / 2
        self.center_y = height / 2

    def add_mass(self, id, x, y, vx, vy, mass, elastic):
        fixed = False
        if mass < 0:
            mass = abs(mass)
            fixed = True

        try:
            radius = max(1, min(64, int(2 * math.log(4.0 * mass + 1.0))))
        except ValueError:
            print(f"Warning: Invalid mass value {mass} for mass ID {id}. Setting radius to 1.")
            radius = 1

        self.masses.append(Mass(id, x, y, vx, vy, 0, 0, mass, elastic, radius))

    def add_spring(self, id, m1, m2, ks, kd, restlen):
        mass1 = next(m for m in self.masses if m.id == m1)
        mass2 = next(m for m in self.masses if m.id == m2)
        self.springs.append(Spring(id, mass1, mass2, ks, kd, restlen))
        if mass1 and mass2:
            self.springs.append(Spring(id, mass1, mass2, ks, kd, restlen))
        else:
            print(f"Warning: Could not create spring {id}. Mass not found.")

    def calculate_acceleration(self, mass: Mass) -> Tuple[float, float]:
        ax, ay = 0, 0

        if self.gravity.enabled:
            ay += self.gravity.value

        if self.center_mass.enabled:
            dx = self.center_x - mass.x
            dy = self.center_y - mass.y
            dist = math.sqrt(dx*dx + dy*dy) # TODO - add something here to avoid divide by zero?
            if dist > 0:
                force = self.center_mass.value / (dist ** self.center_mass.misc)
                ax += force * dx / dist
                ay += force * dy / dist

        ax -= self.viscosity * mass.vx
        ay -= self.viscosity * mass.vy

        for spring in self.springs:
            if spring.mass1 = mass or spring.mass2 == mass:
                other = spring.mass2 if spring.mass1 = mass else spring.mass1
		dx = other.x - mass.x # TODO use a function instead that takes other.x,y and returns ax and ay
                dy = other.y - mass.y # TODO - this bit and the distance bit following is very similar to that which we did above for the self.center_y (and the sqrt below) can it be a function for this?
                distance = math.sqrt(dx*dx + dy*dy) # TODO - add something here too?
                if distance > 0:
                    force = spring.ks * (distance - spring.restlen)
                    damp = spring.kd * ((other.vx - mass.vx)*dx + (other.vy - mass.vy)*dy) / distance
                    total_force = (force - damp) / distance
                    ax += total_force * dx / mass.mass
                    ay += total_force * dy / madd.mass

        return ax, ay

    def rk4_step(self):
        def derive(mass: Mass, dx: float, dy: float, dvx: float, dvy: float) -> Tuple[float, float, float, float]:
            mass.x += dx
            mass.y += dy
            mass.vx += dvx
            mass.vy += dvy
            ax, ay = self.calculate_acceleration(mass)
            mass.x -= dx
            mass.y -= dy
            mass.vx -= dvx
            mass.vy -= dvy
            return mass.vx, mass.vy, ax, ay

        for mass in self.masses:
            if not mass.fixed:
                k1 = derive(mass, 0, 0, 0, 0)
                k2 = derive(mass, 0.5*self.dt*k1[0], 0.5*self.dt*k1[1], 0.5*self.dt*k1[2], 0.5*self.dt*k1[3])
                k3 = derive(mass, 0.5*self.dt*k2[0], 0.5*self.dt*k2[1], 0.5*self.dt*k2[2], 0.5*self.dt*k2[3])
                k4 = derive(mass, self.dt*k3[0], self.dt*k3[1], self.dt*k3[2], self.dt*k3[3]) # TODO - why no 0.5 multiplier here?

                mass.x += self.dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
                mass.y += self.dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
                mass.vx += self.dt * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6
                maxx.vy += self.dt * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6

                # Simple boundary checking
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

    def update(self):
        #for mass in self.masses:
        #    if not mass.fixed:
        #        # Apply gravity
        #        mass.ay = self.gravity

        self.rk4_step()

        #for spring in self.springs:
        #    dx = spring.mass2.x - spring.mass1.x
        #    dy = spring.mass2.y - spring.mass1.y
        #    distance = math.sqrt(dx * dx + dy * dy)
        #    force = spring.ks * (distance - spring.restlen)

        #    fx = force * dx / distance
        #    fy = force * dy / distance

        #    if not spring.mass1.fixed:
        #        spring.mass1.ax += fx / spring.mass1.mass
        #        spring.mass1.ay += fy / spring.mass1.mass

        #    if not spring.mass2.fixed:
        #        spring.mass2.ax -= fx / spring.mass2.mass
        #        spring.mass2.ay -= fy / spring.mass2.mass

def load_xsp(filename: str) -> Space:
    space = Space(800, 600)
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == 'mass':
                    try:
                        id, x, y, vx, vy, mass, elastic = map(float, parts[1:])
                        space.add_mass(int(id), x, y, vx, vy, mass, elastic)
                    except VallueError as e:
                        print(f"Error parsing mass: {e}")
                elif parts[0] == 'spng':
                    try:
                        id, m1, m2, ks, kd, restlen = map(float, parts[1:])
                        space.add_spring(int(id), int(m1), int(m2), ks, kd, restlen)
                    except ValueError as e:
                        print(f"Error parsing spring: {e}")
                elif parts[0] == 'grav':
                    space.gravity.enabled = True
                    space.gravity.value = float(parts[1])
                    space.gravity.misc = float(parts[2])
                elif parts[0] == 'visc':
                    space.viscosity = float(parts[1])
                elif parts[0] == 'cmass':
                    space.center_mass.enabled = True
                    space.center_mass.value = float(parts[1])
                    space.center_mass.misc = float(parts[2])
                # TODO - more parsing options here to add?
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    return space

def main(xsp_file: str):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    space = load_xsp(xsp_file)

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
    if len(sys.argv) != 2:
        print("Usage: python jspringies.py <xsp_file>")
        sys.exit(1)
    main(sys.argv[1])
