from types import FrameType

import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import signal
import sys
import time

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 599
GRID_SIZE = 19
PIXEL_SIZE = WINDOW_SIZE // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

MODEL_PATH = "snake_dqn.pth"

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class SnakeGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Snake Game - DQN")
        self.clock = pygame.time.Clock()

        # DQN parameters
        self.input_size = 11
        self.hidden_size = 256
        self.output_size = 4
        self.learning_rate = 0.001
        self.gamma = 0.99995
        self.epsilon = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.batch_size = 64
        self.memory_size = 50000
        self.empty_percentage = 0

        # Initialize DQN and target network
        self.policy_net = DQN(self.input_size, self.hidden_size, self.output_size)
        self.target_net = DQN(self.input_size, self.hidden_size, self.output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.memory_size)

        self.load_model()
        self.reset_game()

        # Initialize logging variables
        self.start_time = time.time()
        self.net_reward = 0

        # Snake stuff
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2), ((GRID_SIZE // 2) - 1, (GRID_SIZE // 2) - 1)]
        self.direction = RIGHT
        self.food = self.spawn_food()
        self.score = 0
        self.avg_score = 0

    def save_model(self):
        torch.save(self.policy_net.state_dict(), MODEL_PATH)
        print("Model saved.")

    def load_model(self):
        try:
            self.policy_net.load_state_dict(torch.load(MODEL_PATH))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print("Model loaded.")
        except FileNotFoundError:
            print("No saved model found, starting fresh.")

    def reset_game(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2),((GRID_SIZE // 2)-1,(GRID_SIZE // 2)-1)]
        self.direction = RIGHT
        self.food = self.spawn_food()
        self.score = 0
        self.avg_score = 0
        return self.get_state()

    def spawn_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if food not in self.snake:
                return food

    def get_state(self):
        head = self.snake[-1]

        point_l = (head[0] - 1, head[1])
        point_r = (head[0] + 1, head[1])
        point_u = (head[0], head[1] - 1)
        point_d = (head[0], head[1] + 1)

        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        state = [
            # Danger straight
            (dir_r and bool(self.is_collision(point_r))) or
            (dir_l and bool(self.is_collision(point_l))) or
            (dir_u and bool(self.is_collision(point_u))) or
            (dir_d and bool(self.is_collision(point_d))),

            # Danger right
            (dir_u and bool(self.is_collision(point_r))) or
            (dir_d and bool(self.is_collision(point_l))) or
            (dir_l and bool(self.is_collision(point_u))) or
            (dir_r and bool(self.is_collision(point_d))),

            # Danger left
            (dir_d and bool(self.is_collision(point_r))) or
            (dir_u and bool(self.is_collision(point_l))) or
            (dir_r and bool(self.is_collision(point_u))) or
            (dir_l and bool(self.is_collision(point_d))),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1]  # food down
        ]

        return np.array(state, dtype=int)

    def is_collision(self, point):
        # Check border collision
        if (point[0] < 0 or point[0] >= GRID_SIZE or
            point[1] < 0 or point[1] >= GRID_SIZE):
            return "border"
        # Check snake collision
        elif point in self.snake:
            return "snake"
        return False

    def get_action(self, state):
        if random.random() < (self.epsilon - self.epsilon_min):
            return random.randint(0, 3)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def move(self, action):
        # Convert action (0,1,2,3) to direction
        if action == 0:  # LEFT
            self.direction = LEFT
        elif action == 1:  # RIGHT
            self.direction = RIGHT
        elif action ==2: # UP
            self.direction = UP
        else:  # DOWN
            self.direction = DOWN

        head = self.snake[-1]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check for collision and get collision type
        collision = self.is_collision(new_head)
        if collision:
            return collision

        self.snake.append(new_head)

        # Check if food eaten
        if new_head == self.food:
            self.score += 1
            self.food = self.spawn_food()
        else:
            self.snake.pop(0)

        return collision

    def train(self):
        # Check if we have enough samples in memory to perform training
        if len(self.memory) < self.batch_size:
            return

        # Sample random batch of transitions from memory
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch
        batch = list(zip(*transitions))

        # Convert batch arrays to PyTorch tensors
        state_batch = torch.FloatTensor(np.array(batch[0]))  # Current states
        action_batch = torch.LongTensor(batch[1])  # Actions taken
        reward_batch = torch.FloatTensor(batch[2])  # Rewards received
        next_state_batch = torch.FloatTensor(np.array(batch[3]))  # Next states
        done_batch = torch.FloatTensor(batch[4])  # Done flags (1 if episode ended, 0 otherwise)

        # Compute Q values for current states using policy network
        # gather() selects the Q-values corresponding to the actions taken
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute maximum Q values for next states using target network
        # detach() prevents gradients from flowing into the target network
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()

        # Compute expected Q values using Bellman equation:
        # Q = reward + gamma * max(Q_next) * (1 - done)
        # done_batch is used to zero out future rewards when episode is finished
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute MSE loss between current and expected Q values
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        # Perform optimization step
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update network weights

        # Decay epsilon for epsilon-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def draw(self):
        self.screen.fill(BLACK)

        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN,
                           (segment[0] * PIXEL_SIZE, segment[1] * PIXEL_SIZE,
                            PIXEL_SIZE - 2, PIXEL_SIZE - 2))

        # Draw food
        pygame.draw.rect(self.screen, BLUE,
                        (self.food[0] * PIXEL_SIZE, self.food[1] * PIXEL_SIZE,
                         PIXEL_SIZE - 2, PIXEL_SIZE - 2))

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def run(self, training_mode=True):
        episodes = 1000 if training_mode else 100
        target_update = 10
        high_score = 0

        for episode in range(episodes):
            state = self.reset_game()
            total_reward = 0
            steps = 0

            length = len(self.snake)
            self.empty_percentage = (length / (GRID_SIZE ** 2)) * 100

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.save_model()
                        pygame.quit()
                        return

                action = self.get_action(state)
                old_score = self.score

                old_high_score = high_score
                collision = self.move(action)
                reward = 0
                high_score = high_score if high_score > self.score else self.score

                # Replace the existing reward calculation in the run method with:
                if collision:
                    # Base penalty for collision remains similar
                    base_penalty = -10 if collision == "border" else -20
                    score_penalty = self.score * -2
                    potential_penalty = -30 if self.score >= high_score else 0
                    reward = base_penalty + score_penalty + potential_penalty
                elif self.score > old_score:
                    # Calculate the manhattan distance between the starting position and food
                    max_possible_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
                    # Calculate actual path length taken
                    actual_path_length = steps - len(self.snake) + 2  # +2 to account for initial snake length

                    # Reward longer paths more
                    path_efficiency = actual_path_length / max_possible_distance
                    # Give maximum reward when path_efficiency is close to 1.5-2x the minimum path
                    optimal_efficiency = 1.75
                    path_reward = 15 * min(path_efficiency / optimal_efficiency, 1.0)

                    reward = path_reward + 15  # Base reward for getting food
                    if high_score > old_high_score:
                        reward += 20  # Additional reward for new high score
                else:
                    # Small reward for exploring new positions
                    reward = -0.01 if self.score > 10 else -0.05

                next_state = self.get_state()

                if training_mode:
                    self.memory.push(state, action, reward, next_state, bool(collision))
                    self.train()

                state = next_state
                total_reward += reward
                steps += 1

                self.draw()
                self.clock.tick(30)

                if collision:
                    break

            # Update target network
            if episode % target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            print(
                f"Episode: {episode}, Score: {self.score}, Total Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {self.epsilon:.2f}, High score: {high_score:.2f}")

            self.net_reward += total_reward

            if time.time() - self.start_time >= 3600:
                print(f"Net Reward: {self.net_reward}")
                self.start_time = time.time()

def signal_handler(_:int,__:FrameType | None):
    game.save_model()
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    game = SnakeGame()
    game.run(training_mode=True)  # Set to False for evaluation

