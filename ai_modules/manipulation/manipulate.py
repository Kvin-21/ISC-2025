"""Manipulation AI - Q-learning agent for debris triage decisions."""
import numpy as np
import random

# Actions: 0=recycle, 1=repair, 2=discard
RECYCLE = 0
REPAIR = 1
DISCARD = 2


class DebrisTriageEnv:
    """Environment for debris resource management decisions."""
    
    def __init__(self, part_count=10, storage_limit=5):
        self.part_count = part_count
        self.storage_limit = storage_limit
        self.reset()
    
    def reset(self):
        self.parts = np.random.choice([RECYCLE, REPAIR, DISCARD], size=self.part_count)
        self.storage = []
        self.current = 0
        return self._state()
    
    def _state(self):
        part_type = self.parts[self.current] if self.current < self.part_count else -1
        return (part_type, len(self.storage))
    
    def step(self, action):
        done = False
        reward = 0
        part = self.parts[self.current]
        
        # Correct action matches the part type
        if action == part:
            reward = 10
            if action == REPAIR and len(self.storage) < self.storage_limit:
                self.storage.append(part)
        else:
            reward = -5
        
        self.current += 1
        done = self.current >= self.part_count
        
        return self._state(), reward, done


class TriageAgent:
    """Q-learning agent for debris triage."""
    
    def __init__(self, actions=(RECYCLE, REPAIR, DISCARD), learning_rate=0.1, discount=0.9, explore_rate=0.2):
        self.q_table = {}
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = explore_rate
    
    def _q_values(self, state):
        return self.q_table.get(state, np.zeros(len(self.actions)))
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return int(np.argmax(self._q_values(state)))
    
    def learn(self, state, action, reward, next_state, done):
        q = self._q_values(state)
        future = 0 if done else max(self._q_values(next_state))
        q[action] += self.lr * (reward + self.gamma * future - q[action])
        self.q_table[state] = q


def train_agent(episodes=500):
    """Train the triage agent for the specified number of episodes."""
    np.random.seed(42)
    random.seed(42)
    
    env = DebrisTriageEnv()
    agent = TriageAgent()
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total += reward
        if ep % 50 == 0:
            print(f"Episode {ep} Reward: {total}")
    
    return agent


def evaluate_agent(agent, runs=5):
    """Run the agent and print its decisions."""
    env = DebrisTriageEnv()
    for run in range(runs):
        state = env.reset()
        done = False
        total = 0
        actions = []
        while not done:
            action = agent.choose_action(state)
            actions.append(action)
            next_state, reward, done = env.step(action)
            state = next_state
            total += reward
        print(f"Eval {run} Reward: {total}, Actions: {actions}")


if __name__ == "__main__":
    print("Training Manipulation AI Agent...")
    agent = train_agent(episodes=500)
    print("\n--- Evaluation ---")
    evaluate_agent(agent, runs=5)
