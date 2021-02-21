import numpy as np
import torch
from model import make_model

def train(target, predict, epochs, gamma, epsilon):
    actions = 7
    game_over = False

    for i in range(C):
        with torch.no_grad():
            while not game_over:
                q_values = target.forward(board)
                if np.random.uniform() < epsilon:
                    score, action = torch.max(q_values)
                else:
                    action = torch.randperm(torch.size(-1))[0]
                    score = q_values[action]

                new_state, game_over = do_action()
                reward = reward()
                replay = (state, action, reward, new_state)
                buffer.append(replay)

                state = new_state

        batch = [np.random.choice(buffer) for i in range(np.random.randint(1, epochs))]
        for (state, action, reward, new_state) in batch:
            target_q_values = target.forward(new_state)
            prediction_q_values = predict.forward(state)

            target_score, _ = torch.max(target_q_values)
            prediction_score, _ = prediction_q_values[action]

            loss = (reward + gamma * target_score - prediction_score)**2
            loss.backward()

    target = predict.deepcopy()

if __name__ == "__main__":
    if args.train:
        for e in range(args.episodes):
            target = make_model()
            predict = make_model()
            train(target, predict, args.epoch, args.gamma, args.epsilon)
    else:
        play()