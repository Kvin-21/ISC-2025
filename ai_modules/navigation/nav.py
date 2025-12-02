"""Navigation AI for autonomous satellite path planning with collision avoidance."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def create_debris_field(count=10, bounds=100, max_speed=0.5):
    """Generate random debris positions and velocities in 3D space."""
    positions = np.random.rand(count, 3) * bounds
    velocities = (np.random.rand(count, 3) - 0.5) * 2 * max_speed
    return positions, velocities


def compute_avoidance_step(current_pos, goal_pos, obstacles, safe_margin=5.0):
    """Calculate next position, steering away from nearby obstacles."""
    heading = goal_pos - current_pos
    heading = heading / np.linalg.norm(heading)
    
    for obstacle in obstacles:
        offset = obstacle - current_pos
        distance = np.linalg.norm(offset)
        if distance < safe_margin:
            heading -= offset / distance * 0.5

    heading = heading / np.linalg.norm(heading)
    return current_pos + heading * 1.0


def measure_distances(origin, targets):
    """Return distances from origin to each target."""
    return np.linalg.norm(targets - origin, axis=1)

class DebrisField:
    """Manages a collection of debris particles with gravitational drift."""
    
    def __init__(self, count, bounds, max_speed=0.5):
        self.count = count
        self.bounds = bounds
        self.max_speed = max_speed
        self.gravity_focus = np.array([bounds/2, bounds/2, bounds/2])
        self.reset()
    
    def reset(self):
        self.positions = np.random.rand(self.count, 3) * self.bounds
        self.velocities = (np.random.rand(self.count, 3) - 0.5) * 2 * self.max_speed
    
    def gravity_pull(self, pos, strength=0.02):
        """Calculate gravitational acceleration towards focus point."""
        direction = self.gravity_focus - pos
        if pos.ndim == 1:
            dist = np.linalg.norm(direction)
            if dist == 0:
                dist = 1
            return direction / dist * strength
        else:
            dist = np.linalg.norm(direction, axis=1, keepdims=True)
            dist[dist == 0] = 1
            return direction / dist * strength
    
    def tick(self, strength=0.02):
        """Advance debris positions by one timestep."""
        self.velocities += self.gravity_pull(self.positions, strength)
        self.positions += self.velocities
        self.positions = np.mod(self.positions, self.bounds)

class PathPlanner:
    """Generates safe waypoints avoiding obstacles."""
    
    def __init__(self, safe_margin=5.0):
        self.safe_margin = safe_margin
    
    def next_waypoint(self, current_pos, goal_pos, obstacles):
        return compute_avoidance_step(current_pos, goal_pos, obstacles, self.safe_margin)

class Satellite:
    """Represents a satellite navigating through debris to reach a target."""
    
    def __init__(self, start, goal, debris_field, planner, gravity_focus, gravity_strength=0.02):
        self.pos = start.copy()
        self.goal = goal
        self.debris_field = debris_field
        self.planner = planner
        self.gravity_focus = gravity_focus
        self.gravity_strength = gravity_strength
        self.path = [self.pos.copy()]
    
    def step(self):
        """Move one step towards target, balancing avoidance and gravity."""
        waypoint = self.planner.next_waypoint(self.pos, self.goal, self.debris_field.positions) - self.pos
        if np.linalg.norm(waypoint) != 0:
            waypoint = waypoint / np.linalg.norm(waypoint)
        
        pull = self.gravity_focus - self.pos
        pull = pull / np.linalg.norm(pull)
        
        # Blend navigation and gravitational pull
        blend_nav, blend_grav = 0.6, 0.4
        heading = blend_nav * waypoint + blend_grav * pull
        heading = heading / np.linalg.norm(heading)
        
        self.pos += heading * 1.0
        self.path.append(self.pos.copy())
    
    def distance_to_goal(self):
        return np.linalg.norm(self.pos - self.goal)
    
    def has_arrived(self, threshold=1.0):
        return self.distance_to_goal() < threshold