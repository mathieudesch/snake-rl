import torch
import random
import numpy as np
from collections import deque
from snake_rl.environment import SnakeGameAI, Direction, Point
from snake_rl.model import Linear_QNet, QTrainer

MAX_MEMORY = 500_000
BATCH_SIZE = 30
LR = 0.0005
TARGET_UPDATE_FREQUENCY = 25
BLOCK_SIZE = 20

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.99
        self.memory = deque(maxlen=MAX_MEMORY)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Linear_QNet(49, 256, 3).to(self.device)
        self.target_model = Linear_QNet(49, 256, 3).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        grid_size = 7
        state = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create a grid representation of the game state around the snake's head
        for r in range(grid_size):
            for c in range(grid_size):
                # World coordinates for this grid cell
                x = head.x + (c - grid_size // 2) * BLOCK_SIZE
                y = head.y + (r - grid_size // 2) * BLOCK_SIZE
                
                p = Point(x, y)
                
                # 3: Wall
                if x < 0 or x >= game.w or y < 0 or y >= game.h:
                    state[r, c] = 3
                    continue
                
                # 2: Food
                if p == game.food:
                    state[r, c] = 2
                    continue

                # 1: Snake Body
                if p in game.snake:
                    state[r, c] = 1
                    
        return state.flatten()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        if self.n_games > 0 and self.n_games % TARGET_UPDATE_FREQUENCY == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(0.01, 0.999**self.n_games)
        final_move = [0, 0, 0]
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move 