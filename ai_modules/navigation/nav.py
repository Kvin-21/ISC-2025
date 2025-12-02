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

def run_navigation_demo(save_path="navigation_3rd_person.gif"):
    """Run a demonstration of satellite navigation through debris."""
    np.random.seed(42)
    
    debris_count = 10
    bounds = 100
    sat_pos = np.array([0.0, 0.0, 0.0])
    goal = np.array([90.0, 90.0, 90.0])
    gravity_focus = np.array([50.0, 50.0, 50.0])
    grav_strength = 0.02
    
    debris_pos, debris_vel = create_debris_field(count=debris_count, bounds=bounds, max_speed=0.5)
    flight_path = [sat_pos.copy()]
    
    def gravity_pull(pos, focus=gravity_focus, strength=grav_strength):
        direction = focus - pos
        if pos.ndim == 1:
            dist = np.linalg.norm(direction)
            if dist == 0:
                dist = 1
            return direction / dist * strength
        else:
            dist = np.linalg.norm(direction, axis=1, keepdims=True)
            dist[dist == 0] = 1
            return direction / dist * strength
    
    if save_path:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        sat_marker = ax.scatter(*sat_pos, color='blue', label='Satellite')
        goal_marker = ax.scatter(*goal, color='green', label='Target')
        debris_markers = ax.scatter(debris_pos[:, 0], debris_pos[:, 1], debris_pos[:, 2], color='red', label='Debris')
        trail_line, = ax.plot([], [], [], color='cyan', linestyle='--', linewidth=1.5)
        
        ax.set_xlim(0, bounds)
        ax.set_ylim(0, bounds)
        ax.set_zlim(0, bounds)
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.tick_params(colors='white')
        ax.grid(False)
        ax.legend(facecolor='black', edgecolor='white')
        
        def animate(frame):
            nonlocal sat_pos, debris_pos, debris_vel, flight_path
            
            if np.linalg.norm(sat_pos - goal) < 1.0:
                return sat_marker, debris_markers, trail_line
            
            debris_vel += gravity_pull(debris_pos)
            debris_pos += debris_vel
            debris_pos = np.mod(debris_pos, bounds)
            
            waypoint = compute_avoidance_step(sat_pos, goal, debris_pos) - sat_pos
            if np.linalg.norm(waypoint) != 0:
                waypoint = waypoint / np.linalg.norm(waypoint)
            grav_vec = gravity_pull(sat_pos)
            grav_vec = grav_vec / np.linalg.norm(grav_vec)
            
            blend_nav, blend_grav = 0.6, 0.4
            heading = blend_nav * waypoint + blend_grav * grav_vec
            heading = heading / np.linalg.norm(heading)
            sat_pos += heading * 1.0
            flight_path.append(sat_pos.copy())
            
            sat_marker._offsets3d = (sat_pos[0:1], sat_pos[1:2], sat_pos[2:3])
            debris_markers._offsets3d = (debris_pos[:, 0], debris_pos[:, 1], debris_pos[:, 2])
            
            trail = np.array(flight_path)
            trail_line.set_data(trail[:, 0], trail[:, 1])
            trail_line.set_3d_properties(trail[:, 2])
            
            return sat_marker, debris_markers, trail_line
        
        anim = FuncAnimation(fig, animate, frames=500, interval=100, blit=False)
        anim.save(save_path, writer='pillow', fps=10)
        plt.close(fig)
    else:
        for _ in range(500):
            if np.linalg.norm(sat_pos - goal) < 1.0:
                break
            
            debris_vel += gravity_pull(debris_pos)
            debris_pos += debris_vel
            debris_pos = np.mod(debris_pos, bounds)
            
            waypoint = compute_avoidance_step(sat_pos, goal, debris_pos) - sat_pos
            if np.linalg.norm(waypoint) != 0:
                waypoint = waypoint / np.linalg.norm(waypoint)
            grav_vec = gravity_pull(sat_pos)
            grav_vec = grav_vec / np.linalg.norm(grav_vec)
            
            blend_nav, blend_grav = 0.6, 0.4
            heading = blend_nav * waypoint + blend_grav * grav_vec
            heading = heading / np.linalg.norm(heading)
            sat_pos += heading * 1.0
            flight_path.append(sat_pos.copy())
    
    return {
        'trajectory': flight_path,
        'trajectory_length': len(flight_path),
        'final_distance': float(np.linalg.norm(sat_pos - goal)),
        'reached_target': bool(np.linalg.norm(sat_pos - goal) < 1.0),
        'steps': len(flight_path)
    }

if __name__ == "__main__":
    print("Running Navigation AI Demo...")
    save_file = os.path.join(os.path.expanduser("~"), "Downloads", "navigation_3rd_person.gif")
    result = run_navigation_demo(save_path=save_file)
    print(f"Trajectory length: {result['steps']}")
    print(f"Target reached: {result['reached_target']}")
    print(f"Final distance: {result['final_distance']:.2f}")
    print(f"Navigation visualization saved to: {save_file}")
