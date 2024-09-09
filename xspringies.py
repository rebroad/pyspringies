#!/bin/python

import pygame
import sys
import math
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

class Space:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.masses: List[Mass] = []
        self.springs: List[Spring] = []
        self.gravity = 9.8
        self.dt = 0.1

    def add_mass(self, id, x, y, vx, vy, mass, elastic):
        radius = int(2 * math.log(4.0 * mass + 1.0))
        if radius < 1:
            radius = 1
        elif radius > 64:
            radius = 64
        self.masses.append(Mass(id, x, y, vx, vy, 0, 0, mass, elastic, radius))

    def add_spring(self, id, m1, m2, ks, kd, restlen):
        mass1 = next(m for m in self.masses if m.id == m1)
        mass2 = next(m for m in self.masses if m.id == m2)
        self.springs.append(Spring(id, mass1, mass2, ks, kd, restlen))

    def update(self):
        for mass in self.masses:
            if not mass.fixed:
                # Apply gravity
                mass.ay = self.gravity

                # Update position and velocity using Euler integration
                mass.vx += mass.ax * self.dt
                mass.vy += mass.ay * self.dt
                mass.x += mass.vx * self.dt
                mass.y += mass.vy * self.dt

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

        for spring in self.springs:
            dx = spring.mass2.x - spring.mass1.x
            dy = spring.mass2.y - spring.mass1.y
            distance = math.sqrt(dx * dx + dy * dy)
            force = spring.ks * (distance - spring.restlen)

            fx = force * dx / distance
            fy = force * dy / distance

            if not spring.mass1.fixed:
                spring.mass1.ax += fx / spring.mass1.mass
                spring.mass1.ay += fy / spring.mass1.mass

            if not spring.mass2.fixed:
                spring.mass2.ax -= fx / spring.mass2.mass
                spring.mass2.ay -= fy / spring.mass2.mass

def load_xsp(filename: str) -> Space:
    space = Space(800, 600)
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'mass':
                id, x, y, vx, vy, mass, elastic = map(float, parts[1:])
                space.add_mass(int(id), x, y, vx, vy, mass, elastic)
            elif parts[0] == 'spng':
                id, m1, m2, ks, kd, restlen = map(float, parts[1:])
                space.add_spring(int(id), int(m1), int(m2), ks, kd, restlen)
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
