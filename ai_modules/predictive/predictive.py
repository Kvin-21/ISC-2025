"""Predictive Orbit AI - orbital mechanics, collision prediction, and fuel optimisation."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# Physical constants
G = 6.67430e-11  # gravitational constant
EARTH_MASS = 5.972e24
EARTH_RADIUS = 6.371e6
MU = G * EARTH_MASS


class OrbitalMechanics:
    """Handles orbital state propagation and atmospheric drag."""
    
    def __init__(self):
        self.G = G
        self.M_earth = EARTH_MASS
        self.R_earth = EARTH_RADIUS
        self.mu = MU
    
    def elements_to_state(self, semi_major, eccentricity, inclination, arg_peri, raan, true_anomaly):
        """Convert Keplerian elements to position and velocity vectors."""
        r = semi_major * (1 - eccentricity**2) / (1 + eccentricity * np.cos(true_anomaly))
        x_orb = r * np.cos(true_anomaly)
        y_orb = r * np.sin(true_anomaly)
        
        h = np.sqrt(self.mu * semi_major * (1 - eccentricity**2))
        vx_orb = -self.mu / h * np.sin(true_anomaly)
        vy_orb = self.mu / h * (eccentricity + np.cos(true_anomaly))
        
        # Rotation matrices
        rot_peri = np.array([
            [np.cos(arg_peri), -np.sin(arg_peri), 0],
            [np.sin(arg_peri), np.cos(arg_peri), 0],
            [0, 0, 1]
        ])
        rot_inc = np.array([
            [1, 0, 0],
            [0, np.cos(inclination), -np.sin(inclination)],
            [0, np.sin(inclination), np.cos(inclination)]
        ])
        rot_raan = np.array([
            [np.cos(raan), -np.sin(raan), 0],
            [np.sin(raan), np.cos(raan), 0],
            [0, 0, 1]
        ])
        
        rotation = rot_raan @ rot_inc @ rot_peri
        position = rotation @ np.array([x_orb, y_orb, 0])
        velocity = rotation @ np.array([vx_orb, vy_orb, 0])
        
        return position, velocity
    
    def atmospheric_drag(self, position, velocity, altitude_km):
        """Compute drag acceleration using an exponential density model."""
        # Density varies by altitude band
        if altitude_km > 1000:
            rho = 0
        elif altitude_km > 600:
            rho = 1e-15 * np.exp(-(altitude_km - 600) / 100)
        elif altitude_km > 400:
            rho = 1e-12 * np.exp(-(altitude_km - 400) / 100)
        elif altitude_km > 300:
            rho = 1e-11 * np.exp(-(altitude_km - 300) / 100)
        elif altitude_km > 200:
            rho = 1e-10 * np.exp(-(altitude_km - 200) / 100)
        else:
            rho = 1.225 * np.exp(-altitude_km / 8.5)
        
        drag_coeff = 2.2
        area = 10.0
        mass = 500.0
        
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(3)
        
        drag_mag = -0.5 * rho * drag_coeff * area / mass * speed**2
        return drag_mag * (velocity / speed)
    
    def propagate(self, position, velocity, time_step, duration):
        """Propagate orbit using RK4 integration."""
        trajectory = [position.copy()]
        velocities = [velocity.copy()]
        times = [0]
        
        pos = position.copy()
        vel = velocity.copy()
        
        dt = min(time_step, 10)  # smaller step for accuracy
        steps = int(duration / dt)
        save_every = max(1, int(time_step / dt))
        
        for step in range(steps):
            alt = (np.linalg.norm(pos) - self.R_earth) / 1000
            r1 = np.linalg.norm(pos)
            
            # k1
            accel1 = -self.mu / r1**3 * pos + self.atmospheric_drag(pos, vel, alt)
            k1_v = accel1 * dt
            k1_r = vel * dt
            
            # k2
            pos2 = pos + 0.5 * k1_r
            vel2 = vel + 0.5 * k1_v
            alt2 = (np.linalg.norm(pos2) - self.R_earth) / 1000
            r2 = np.linalg.norm(pos2)
            accel2 = -self.mu / r2**3 * pos2 + self.atmospheric_drag(pos2, vel2, alt2)
            k2_v = accel2 * dt
            k2_r = vel2 * dt
            
            # k3
            pos3 = pos + 0.5 * k2_r
            vel3 = vel + 0.5 * k2_v
            alt3 = (np.linalg.norm(pos3) - self.R_earth) / 1000
            r3 = np.linalg.norm(pos3)
            accel3 = -self.mu / r3**3 * pos3 + self.atmospheric_drag(pos3, vel3, alt3)
            k3_v = accel3 * dt
            k3_r = vel3 * dt
            
            # k4
            pos4 = pos + k3_r
            vel4 = vel + k3_v
            alt4 = (np.linalg.norm(pos4) - self.R_earth) / 1000
            r4 = np.linalg.norm(pos4)
            accel4 = -self.mu / r4**3 * pos4 + self.atmospheric_drag(pos4, vel4, alt4)
            k4_v = accel4 * dt
            k4_r = vel4 * dt
            
            pos += (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6
            vel += (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
            
            if (step + 1) % save_every == 0:
                trajectory.append(pos.copy())
                velocities.append(vel.copy())
                times.append((step + 1) * dt)
        
        return np.array(trajectory), np.array(velocities), np.array(times)
    
    # Backwards compatibility alias
    def orbital_elements_to_state(self, a, e, i, omega, Omega, nu):
        return self.elements_to_state(a, e, i, omega, Omega, nu)
    
    def propagate_orbit(self, position, velocity, time_step, duration):
        return self.propagate(position, velocity, time_step, duration)


class CollisionPredictor:
    """ML model for assessing collision probability between objects."""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _generate_data(self, sample_count=10000):
        """Create synthetic training data for the classifier."""
        print("Generating training data for collision prediction...")
        features, labels = [], []
        
        for _ in range(sample_count):
            rel_vel = np.random.uniform(0.1, 15.0)
            miss_dist = np.random.uniform(0, 5.0)
            time_tca = np.random.uniform(0, 3600)
            size1 = np.random.uniform(0.1, 10.0)
            size2 = np.random.uniform(0.1, 10.0)
            inc_diff = np.random.uniform(0, 30)
            alt_diff = np.random.uniform(0, 100)
            uncertainty = np.random.uniform(0.01, 1.0)
            
            threshold = (size1 + size2) / 2000 + uncertainty
            if miss_dist < threshold:
                label = 1
            elif miss_dist < threshold * 2:
                label = np.random.choice([0, 1], p=[0.7, 0.3])
            else:
                label = 0
            
            features.append([rel_vel, miss_dist, time_tca, size1, size2, inc_diff, alt_diff, uncertainty])
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def train(self, X=None, y=None):
        """Fit the collision predictor."""
        if X is None or y is None:
            X, y = self._generate_data()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training collision risk model...")
        self.model.fit(X_train_scaled, y_train)
        print(f"Training accuracy: {self.model.score(X_train_scaled, y_train):.4f}")
        print(f"Testing accuracy: {self.model.score(X_test_scaled, y_test):.4f}")
        self.is_trained = True
    
    def predict(self, features):
        """Return collision probability for given features."""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        return self.model.predict_proba(self.scaler.transform([features]))[0][1]
    
    # Backwards compatibility
    def predict_collision_risk(self, features):
        return self.predict(features)
    
    def generate_training_data(self, n_samples=10000):
        return self._generate_data(n_samples)
    
    def save_model(self, filepath='collision_model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'is_trained': self.is_trained}, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='collision_model.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model, self.scaler, self.is_trained = data['model'], data['scaler'], data['is_trained']
        print(f"Model loaded from {filepath}")


# Backwards compatibility alias
CollisionRiskPredictor = CollisionPredictor


class FuelOptimiser:
    """Calculates delta-v and fuel requirements for orbital manoeuvres."""
    
    def __init__(self):
        self.mu = 3.986004418e14  # Earth's gravitational parameter
        self.isp = 300  # specific impulse (s)
        self.g0 = 9.80665  # standard gravity
    
    def delta_v(self, r1, r2):
        """Compute Hohmann transfer delta-v between two circular orbits."""
        v1 = np.sqrt(self.mu / r1)
        v2 = np.sqrt(self.mu / r2)
        a_transfer = (r1 + r2) / 2
        v_t1 = np.sqrt(self.mu * (2/r1 - 1/a_transfer))
        v_t2 = np.sqrt(self.mu * (2/r2 - 1/a_transfer))
        total = abs(v_t1 - v1) + abs(v2 - v_t2)
        return total, abs(v_t1 - v1), abs(v2 - v_t2)
    
    # Backwards compatibility
    def calculate_delta_v(self, r1, r2):
        return self.delta_v(r1, r2)
    
    def fuel_mass(self, dv, dry_mass):
        """Tsiolkovsky equation for propellant mass."""
        return dry_mass * (np.exp(dv / (self.isp * self.g0)) - 1)
    
    # Backwards compatibility
    def calculate_fuel_mass(self, dv, dry_mass):
        return self.fuel_mass(dv, dry_mass)
    
    def plan_manoeuvres(self, manoeuvres, available_fuel):
        """Select manoeuvres by urgency while respecting fuel budget."""
        sequence = []
        used = 0
        for m in sorted(manoeuvres, key=lambda x: x['urgency'], reverse=True):
            fuel_needed = self.fuel_mass(m['delta_v'], 500)
            if used + fuel_needed <= available_fuel:
                sequence.append(m)
                used += fuel_needed
            else:
                print(f"⚠️ Insufficient fuel for {m['name']}")
        return sequence, used
    
    # Backwards compatibility
    def optimize_maneuver_sequence(self, maneuvers, fuel_avail):
        return self.plan_manoeuvres(maneuvers, fuel_avail)


# Backwards compatibility alias
FuelOptimizer = FuelOptimiser


class AlertSystem:
    """Collects and prioritises mission alerts."""
    
    URGENCY_ORDER = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2}
    
    def __init__(self):
        self.alerts = []
    
    def _add(self, alert_type, urgency, message):
        self.alerts.append({
            'type': alert_type,
            'urgency': urgency,
            'message': message,
            'timestamp': datetime.now()
        })
    
    def check_collision_risk(self, probability, time_to_closest):
        if probability > 0.7:
            urgency, icon = "CRITICAL", "🔴"
        elif probability > 0.4:
            urgency, icon = "HIGH", "🟡"
        elif probability > 0.2:
            urgency, icon = "MEDIUM", "🟠"
        else:
            return None
        
        msg = f"{icon} {urgency} collision risk: {probability:.1%} probability in {time_to_closest:.0f}s"
        self._add('COLLISION_RISK', urgency, msg)
    
    def check_orbital_decay(self, altitude_km, decay_rate_km_day):
        critical_alt = 150
        if decay_rate_km_day >= 0:
            return None
        
        days_to_reentry = (altitude_km - critical_alt) / abs(decay_rate_km_day)
        
        if days_to_reentry < 30:
            urgency, icon = "CRITICAL", "🔴"
        elif days_to_reentry < 90:
            urgency, icon = "HIGH", "🟡"
        elif days_to_reentry < 365:
            urgency, icon = "MEDIUM", "🟠"
        else:
            return None
        
        alert = {
            'type': 'ORBITAL_DECAY',
            'urgency': urgency,
            'current_altitude': altitude_km,
            'decay_rate': decay_rate_km_day,
            'time_to_reentry_days': days_to_reentry,
            'message': f"{icon} {urgency} orbital decay: {altitude_km:.1f} km altitude, reentry in {days_to_reentry:.0f} days",
            'timestamp': datetime.now()
        }
        self.alerts.append(alert)
        return alert
    
    def check_fuel_level(self, remaining, capacity, required):
        pct = remaining / capacity * 100
        if remaining < required:
            urgency, icon = "CRITICAL", "🔴"
            msg = f"{icon} {urgency}: Insufficient fuel! Need {required:.1f}kg, have {remaining:.1f}kg"
        elif pct < 20:
            urgency, icon = "HIGH", "🟡"
            msg = f"{icon} {urgency}: Fuel low at {pct:.1f}%"
        elif pct < 40:
            urgency, icon = "MEDIUM", "🟠"
            msg = f"{icon} {urgency}: Fuel at {pct:.1f}%"
        else:
            return None
        
        self._add('FUEL_LOW', urgency, msg)
    
    def get_all_alerts(self):
        return sorted(self.alerts, key=lambda a: self.URGENCY_ORDER[a['urgency']])
    
    def print_alerts(self):
        print("\n" + "=" * 60 + "\n🚨 ALERT SYSTEM REPORT\n" + "=" * 60)
        if not self.alerts:
            print("✅ No alerts. All systems nominal.")
        else:
            for i, alert in enumerate(self.get_all_alerts(), 1):
                print(f"\n[{i}] {alert['message']}")
                print(f"    Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60 + "\n")


class PredictiveOrbitAI:
    """Orchestrates orbit propagation, collision analysis, and fuel planning."""
    
    def __init__(self):
        self.orbital_mech = OrbitalMechanics()
        self.collision_predictor = CollisionPredictor()
        self.fuel_optimiser = FuelOptimiser()
        self.alert_system = AlertSystem()
        self.collision_predictor.train()
        # Backwards compatibility alias
        self.fuel_optimizer = self.fuel_optimiser
    
    def analyse(self, sat_params, debris_list, duration=86400):
        """Run full mission analysis."""
        print("\n🛰️ Starting Predictive Orbit AI Analysis...\n" + "=" * 60)
        
        print("\n[1/4] Propagating satellite orbit...")
        pos, vel = self.orbital_mech.orbital_elements_to_state(
            sat_params['semi_major_axis'], sat_params['eccentricity'],
            sat_params['inclination'], sat_params['arg_periapsis'],
            sat_params['raan'], sat_params['true_anomaly']
        )
        
        traj, vels, times = self.orbital_mech.propagate_orbit(pos, vel, 60, duration)
        
        initial_alt = np.linalg.norm(traj[0]) - self.orbital_mech.R_earth
        final_alt = np.linalg.norm(traj[-1]) - self.orbital_mech.R_earth
        decay_rate = (final_alt - initial_alt) / duration * 86400 / 1000
        
        print(f"   Initial altitude: {initial_alt/1000:.2f} km")
        print(f"   Final altitude: {final_alt/1000:.2f} km")
        print(f"   Decay rate: {decay_rate:.4f} km/day")
        
        self.alert_system.check_orbital_decay(final_alt/1000, decay_rate)
        
        print("\n[2/4] Analysing collision risks with debris...")
        risks = []
        for debris in debris_list:
            d_pos, d_vel = self.orbital_mech.orbital_elements_to_state(
                debris['semi_major_axis'], debris['eccentricity'], debris['inclination'],
                debris['arg_periapsis'], debris['raan'], debris['true_anomaly']
            )
            
            rel_vel = np.linalg.norm(vel - d_vel) / 1000
            miss_dist = np.linalg.norm(pos - d_pos) / 1000
            time_closest = miss_dist / rel_vel if rel_vel > 0 else float('inf')
            
            features = [
                rel_vel, miss_dist, time_closest,
                sat_params['size'], debris['size'],
                abs(sat_params['inclination'] - debris['inclination']) * 180 / np.pi,
                abs(sat_params['altitude'] - debris['altitude']),
                debris.get('position_uncertainty', 0.1)
            ]
            
            probability = self.collision_predictor.predict_collision_risk(features)
            risks.append({
                'debris_id': debris['id'],
                'probability': probability,
                'miss_distance': miss_dist,
                'relative_velocity': rel_vel,
                'time_to_closest': time_closest
            })
            self.alert_system.check_collision_risk(probability, time_closest)
        
        risks.sort(key=lambda r: r['probability'], reverse=True)
        print(f"   Analysed {len(risks)} potential conjunctions")
        print(f"   Highest risk: {risks[0]['probability']:.1%}")
        
        print("\n[3/4] Optimising fuel usage...")
        manoeuvres = []
        for risk in risks:
            if risk['probability'] > 0.3:
                manoeuvres.append({
                    'name': f"Avoid debris {risk['debris_id']}",
                    'delta_v': risk['relative_velocity'] * 100,
                    'urgency': risk['probability'] * 10
                })
        
        if decay_rate < -0.1:
            dv, _, _ = self.fuel_optimiser.calculate_delta_v(
                np.linalg.norm(traj[-1]), sat_params['semi_major_axis']
            )
            manoeuvres.append({'name': "Orbit raising manoeuvre", 'delta_v': dv, 'urgency': 7})
        
        fuel_available = sat_params.get('fuel_mass', 50)
        fuel_capacity = sat_params.get('fuel_capacity', 100)
        planned, fuel_needed = self.fuel_optimiser.plan_manoeuvres(manoeuvres, fuel_available)
        
        print(f"   Planned manoeuvres: {len(planned)}")
        print(f"   Total fuel needed: {fuel_needed:.2f} kg")
        print(f"   Fuel available: {fuel_available:.2f} kg")
        
        self.alert_system.check_fuel_level(fuel_available, fuel_capacity, fuel_needed)
        
        print("\n[4/4] Generating alerts...")
        self.alert_system.print_alerts()
        
        report = {
            'orbital_analysis': {
                'initial_altitude_km': initial_alt / 1000,
                'final_altitude_km': final_alt / 1000,
                'decay_rate_km_per_day': decay_rate,
                'trajectory': traj,
                'times': times
            },
            'collision_analysis': {
                'total_conjunctions': len(risks),
                'high_risk_count': sum(1 for r in risks if r['probability'] > 0.5),
                'risks': risks[:10]
            },
            'fuel_analysis': {
                'fuel_available': fuel_available,
                'fuel_needed': fuel_needed,
                'fuel_remaining_after': fuel_available - fuel_needed,
                'maneuvers': planned
            },
            'alerts': self.alert_system.get_all_alerts()
        }
        
        print("\n Analysis complete!")
        return report
    
    # Backwards compatibility
    def analyze_mission(self, sat_params, debris_list, duration=86400):
        return self.analyse(sat_params, debris_list, duration)
    
    def visualise(self, report, save_path=None):
        """Create a 6-panel analysis visualisation."""
        fig = plt.figure(figsize=(16, 12))
        
        # 3D trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        traj = report['orbital_analysis']['trajectory']
        ax1.plot(traj[:, 0]/1e6, traj[:, 1]/1e6, traj[:, 2]/1e6, 'b-', linewidth=2, label='Satellite')
        u, v = np.linspace(0, 2*np.pi, 50), np.linspace(0, np.pi, 50)
        x = 6.371 * np.outer(np.cos(u), np.sin(v))
        y = 6.371 * np.outer(np.sin(u), np.sin(v))
        z = 6.371 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x, y, z, color='blue', alpha=0.3)
        ax1.set_xlabel('X (1000 km)')
        ax1.set_ylabel('Y (1000 km)')
        ax1.set_zlabel('Z (1000 km)')
        ax1.set_title('Orbital Trajectory Prediction')
        ax1.legend()
        
        # Altitude decay
        ax2 = fig.add_subplot(2, 3, 2)
        times = report['orbital_analysis']['times'] / 3600
        altitudes = [np.linalg.norm(p)/1000 - 6371 for p in traj]
        ax2.plot(times, altitudes, 'r-', linewidth=2)
        ax2.axhline(y=150, color='orange', linestyle='--', label='Critical altitude')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Altitude (km)')
        ax2.set_title('Orbital Decay Prediction')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Collision risks
        ax3 = fig.add_subplot(2, 3, 3)
        probs = [r['probability'] for r in report['collision_analysis']['risks']]
        colours = ['red' if p > 0.5 else 'orange' if p > 0.3 else 'yellow' for p in probs]
        ax3.bar(range(len(probs)), probs, color=colours)
        ax3.set_xlabel('Debris')
        ax3.set_ylabel('Collision Probability')
        ax3.set_title('Top 10 Collision Risks')
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Fuel budget
        ax4 = fig.add_subplot(2, 3, 4)
        manoeuvres = report['fuel_analysis']['maneuvers']
        if manoeuvres:
            names = [m['name'][:20] for m in manoeuvres]
            fuel = [self.fuel_optimiser.fuel_mass(m['delta_v'], 500) for m in manoeuvres]
            ax4.barh(names, fuel, color='green')
            ax4.set_xlabel('Fuel Required (kg)')
            ax4.grid(True, alpha=0.3, axis='x')
        else:
            ax4.text(0.5, 0.5, 'No manoeuvres planned', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Fuel Budget per Manoeuvre')
        
        # Alert summary
        ax5 = fig.add_subplot(2, 3, 5)
        alerts = report['alerts']
        if alerts:
            categories = {}
            for a in alerts:
                key = f"{a['type']}\n({a['urgency']})"
                categories[key] = categories.get(key, 0) + 1
            palette = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow'}
            colours = [palette.get(k.split('\n')[1].strip('()'), 'gray') for k in categories]
            ax5.bar(range(len(categories)), categories.values(), color=colours, tick_label=list(categories.keys()))
            ax5.set_ylabel('Number of Alerts')
            ax5.grid(True, alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, ' No alerts', ha='center', va='center', transform=ax5.transAxes, fontsize=16)
        ax5.set_title('Alert Summary')
        
        # Timeline
        ax6 = fig.add_subplot(2, 3, 6)
        events = []
        for r in report['collision_analysis']['risks'][:5]:
            if r['time_to_closest'] < 86400:
                events.append({
                    'time': r['time_to_closest'] / 3600,
                    'event': f"Debris {r['debris_id']}",
                    'type': 'collision'
                })
        for i, m in enumerate(manoeuvres):
            events.append({
                'time': (i + 1) * 24 / (len(manoeuvres) + 1),
                'event': m['name'][:25],
                'type': 'maneuver'
            })
        events.sort(key=lambda e: e['time'])
        
        if events:
            for i, e in enumerate(events):
                colour = 'red' if e['type'] == 'collision' else 'green'
                marker = 'o' if e['type'] == 'collision' else 's'
                ax6.scatter(e['time'], i, color=colour, marker=marker, s=100, zorder=3)
                ax6.text(e['time'] + 0.5, i, e['event'], va='center', fontsize=8)
            ax6.set_xlabel('Time (hours)')
            ax6.set_ylabel('Event')
            ax6.set_yticks(range(len(events)))
            ax6.set_yticklabels([])
            ax6.grid(True, alpha=0.3, axis='x')
            ax6.set_xlim(0, 24)
        else:
            ax6.text(0.5, 0.5, 'No events in next 24h', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Mission Timeline (Next 24h)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n Visualisation saved to: {save_path}")
        plt.show()
    
    # Backwards compatibility
    def visualize_predictions(self, report, save_path=None):
        return self.visualise(report, save_path)


def run_example_mission():
    """Demonstrate the predictive orbit AI with sample data."""
    predictor = PredictiveOrbitAI()
    
    satellite = {
        'semi_major_axis': EARTH_RADIUS + 400e3,
        'eccentricity': 0.0,
        'inclination': np.deg2rad(51.6),
        'arg_periapsis': 0,
        'raan': 0,
        'true_anomaly': 0,
        'altitude': 400,
        'size': 5.0,
        'fuel_mass': 5,
        'fuel_capacity': 100
    }
    
    np.random.seed(42)
    debris_list = []
    for i in range(15):
        alt = np.random.uniform(398, 402)  # close to satellite
        debris_list.append({
            'id': f'DEBRIS-{i+1:03d}',
            'semi_major_axis': EARTH_RADIUS + alt * 1e3,
            'eccentricity': np.random.uniform(0.0001, 0.002),
            'inclination': np.deg2rad(np.random.uniform(50, 53)),
            'arg_periapsis': np.random.uniform(0, 2*np.pi),
            'raan': np.random.uniform(0, 2*np.pi),
            'true_anomaly': np.random.uniform(0, 2*np.pi),
            'altitude': alt,
            'size': np.random.uniform(1.0, 5.0),
            'position_uncertainty': np.random.uniform(0.2, 0.5)
        })
    
    report = predictor.analyze_mission(satellite, debris_list, duration=86400)
    
    output_path = os.path.join(os.path.expanduser("~"), "Downloads", "predictive_orbit_ai_analysis.png")
    predictor.visualize_predictions(report, save_path=output_path)
    
    # Summary
    print("\n" + "=" * 80 + "\n📋 MISSION ANALYSIS SUMMARY\n" + "=" * 80)
    orb = report['orbital_analysis']
    coll = report['collision_analysis']
    fuel = report['fuel_analysis']
    
    print(f"\n ORBITAL ANALYSIS:")
    print(f"   Current altitude: {orb['final_altitude_km']:.2f} km")
    print(f"   Decay rate: {orb['decay_rate_km_per_day']:.4f} km/day")
    
    print(f"\n COLLISION ANALYSIS:")
    print(f"   Total conjunctions: {coll['total_conjunctions']}")
    print(f"   High-risk: {coll['high_risk_count']}")
    if coll['risks']:
        top = coll['risks'][0]
        print(f"   Highest risk: {top['probability']:.1%} with {top['debris_id']}")
        print(f"   Miss distance: {top['miss_distance']:.2f} km")
    
    print(f"\n FUEL ANALYSIS:")
    print(f"   Available: {fuel['fuel_available']:.2f} kg")
    print(f"   Needed: {fuel['fuel_needed']:.2f} kg")
    print(f"   Remaining: {fuel['fuel_remaining_after']:.2f} kg")
    print(f"   Manoeuvres: {len(fuel['maneuvers'])}")
    
    print(f"\n ALERTS:")
    print(f"   Total: {len(report['alerts'])}")
    critical = [a for a in report['alerts'] if a['urgency'] == 'CRITICAL']
    if critical:
        print(f"   ⚠️ CRITICAL: {len(critical)}")
        for a in critical:
            print(f"      - {a['message']}")
    
    print("\n" + "=" * 80 + "\n DEMONSTRATION COMPLETE\n" + "=" * 80)
    
    predictor.collision_predictor.save_model('collision_model.pkl')
    print("\n Model saved to collision_model.pkl")
    
    return report, predictor


if __name__ == "__main__":
    run_example_mission()
