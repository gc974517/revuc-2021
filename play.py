import numpy as np
import torch
import torch.optim as optim
import pygame
import argparse
import random
from copy import deepcopy
from model import make_model
from tetris import run

parser = argparse.ArgumentParser()
parser.add_argument('-train',
                    action='store_true')
parser.add_argument('-episodes',
                    type=int,
                    default=100)
parser.add_argument('-epochs',
                    type=int,
                    default=100,)
parser.add_argument('-gamma',
                    type=float,
                    default=0.9)
parser.add_argument('-epsilon',
                    type=float,
                    default=0.9)

args, _ = parser.parse_known_args()

def move(action):
    if action == 0: # Rotate
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)
    elif action == 1: # Down
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN)
    elif action == 2: # Left
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)
    elif action == 3: # Right
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT)
    elif action == 4: # Drop
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)

    return event

def train(target, predict, epochs, gamma, epsilon):
    buffer = []

    game, game_over = run(None)
    state = game.environment

    predict.train()
    target.eval()
    optimizer = optim.RMSprop(predict.parameters())

    for i in range(10):
        while not game_over:
            q_values = target.forward(state)
            if np.random.uniform() < epsilon:
                score, action = torch.max(q_values, dim=-1)
            else:
                action = torch.randperm(q_values.size(-1))[0].item()
                score = q_values[0, action]

            event = move(action)
            game, game_over = run(game, event)

            new_state = game.environment

            reward = 0
            adjacency_scores = []
            for i in range(len(new_state)):
                row = new_state[i]
                # islands = np.where(np.roll(row, 1) != row)[0]
                # print(islands)

                # for i in range(len(islands)):
                #     if i == 0:
                #         adjacency_scores.append(np.sum(row[:islands[i]]))
                #     else:
                #         adjacency_scores.append(np.sum(row[islands[i-1]:islands[i]]))
                adjacency_scores.append(np.sum(row))

            if len(adjacency_scores) > 0:
                reward += np.max(adjacency_scores) * np.argmax(adjacency_scores)
            reward *= (game.score + 1)
            reward -= np.sum(np.expand_dims((10 - np.arange(5)), axis=1) * new_state[:5])
            print(reward)
            # print(new_state)
            replay = (state, action, reward, new_state)
            buffer.append(replay)

            state = new_state

            target_q_values = target.forward(new_state)
            prediction_q_values = predict.forward(state)

            target_score, _ = torch.max(target_q_values, dim=-1)
            prediction_score = prediction_q_values[0, action]

            delta = prediction_score - reward - gamma * target_score
            L_delta = 0.5 * delta**2 if abs(delta) <= 1 else abs(delta) - 1

            loss = L_delta
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch = random.sample(buffer, np.random.randint(1, len(buffer)))
        loss = 0
        for (state, action, reward, new_state) in batch:
            target_q_values = target.forward(new_state)
            prediction_q_values = predict.forward(state)

            target_score, _ = torch.max(target_q_values, dim=-1)
            prediction_score = prediction_q_values[0, action]

            delta = prediction_score - reward - gamma * target_score
            L_delta = 0.5 * delta**2 if abs(delta) <= 1 else abs(delta) - 1

            loss += L_delta
        
        loss /= len(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    target = deepcopy(predict)
    torch.save(target, "model.pt")

if __name__ == "__main__":
    board_size = 100
    hidden_size = 64
    state_size = 64
    actions = 5
    layers = 1

    if args.train:
        while True:
            target = make_model(board_size, hidden_size, state_size, actions, layers)
            predict = make_model(board_size, hidden_size, state_size, actions, layers)
            train(target, predict, args.epochs, args.gamma, args.epsilon)
    else:
        play()