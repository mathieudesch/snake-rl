import torch
from snake_rl.agent import Agent
from snake_rl.environment import SnakeGameAI
from utils import plot
import time
import math
import pygame
import datetime
import os

NUM_ENVS = 12
GAME_WIDTH = 320
GAME_HEIGHT = 240
TRAIN_EVERY_N_STEPS = 1024 # Train after this many steps across all envs
PLOT_EVERY_N_GAMES = 12  # Plot every 12 games to reduce lag

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"training_log_{start_time_str}.txt")
    last_log_time = time.time()
    with open(log_file, "w") as f:
        f.write("Timestamp,Games,Record,Epsilon,Mean Score,Mean Score (Last 50),Memory\n")
    
    cols = int(math.sqrt(NUM_ENVS))
    rows = math.ceil(NUM_ENVS / cols)
    
    main_display_width = cols * GAME_WIDTH
    main_display_height = rows * GAME_HEIGHT
    
    pygame.init()
    main_display = pygame.display.set_mode((main_display_width, main_display_height))
    pygame.display.set_caption('Snake RL - All Environments')
    clock = pygame.time.Clock()

    games = []
    for i in range(NUM_ENVS):
        row = i // cols
        col = i % cols
        offset_x = col * GAME_WIDTH
        offset_y = row * GAME_HEIGHT
        surface = main_display.subsurface((offset_x, offset_y, GAME_WIDTH, GAME_HEIGHT))
        game = SnakeGameAI(w=GAME_WIDTH, h=GAME_HEIGHT, surface=surface)
        games.append(game)

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    total_steps = 0
    while True:
        current_time = time.time()
        if current_time - last_log_time > 60:
            last_log_time = current_time
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            games_played = agent.n_games
            record_score = record
            epsilon_val = agent.epsilon
            mean_score_total = total_score / agent.n_games if agent.n_games > 0 else 0.0
            
            last_50_scores = plot_scores[-50:]
            mean_score_50 = sum(last_50_scores) / len(last_50_scores) if last_50_scores else 0.0
            
            memory_usage = f"{len(agent.memory)}/{agent.memory.maxlen}"

            log_line = f"{timestamp},{games_played},{record_score},{epsilon_val:.4f},{mean_score_total:.2f},{mean_score_50:.2f},{memory_usage}\n"
            with open(log_file, "a") as f:
                f.write(log_line)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        for i, game in enumerate(games):
            total_steps += 1
            
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
    
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if total_steps % TRAIN_EVERY_N_STEPS == 0:
                agent.train_long_memory()

            if done:
                game.reset()
                agent.n_games += 1
                
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                
                if agent.n_games % PLOT_EVERY_N_GAMES == 0:
                    plot(plot_scores, plot_mean_scores)

                if score > record:
                    record = score
                    agent.model.save()

                print(f'Game: {agent.n_games}, Env: {i+1}, Score: {score}, Record: {record}')


        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    train() 