from environment import Environment
import pygame
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc
#from memory_profiler import profile
import csv

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIMENSION, NEURONS)
        self.fc2 = nn.Linear(NEURONS, NEURONS)
        self.fc3 = nn.Linear(NEURONS, OUTPUT_DIMENSION)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim = -1)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(INPUT_DIMENSION, NEURONS)
        self.fc2 = nn.Linear(NEURONS, NEURONS)
        self.fc3 = nn.Linear(NEURONS, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) 

def main():
    env = Environment()
    #policy_nn = PolicyNetwork()
    #value_nn = ValueNetwork()

    ppath, vpath, data_path = getPath()
    policynn , _ = load(ppath, vpath)
    #training_loop(env, policy_nn, value_nn, ppath, vpath, data_path)
    
    play(env, policynn)

def getPath():
    path = "v" + str(VERSION) + "\\"
    policy_path = path + "pm.pt"
    value_path = path + "vm.pt"
    data_path = path + "data.csv"

    return policy_path, value_path, data_path

def write_model_training_data(out_file, sizes, policy_losses, value_losses):
    data = []
    for i in range(len(sizes)):
        data.append((sizes[i], policy_losses[i], value_losses[i]))
    

    with open(out_file, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(data)

def training_loop(env, policy_nn, value_nn, policy_path, value_path, data_path):
    
    for i in range(10):
        sizes, policy_Losses, value_Losses = train(env, policy_nn, value_nn, 600, i+1)
        save(policy_nn, value_nn, policy_path, value_path)
        write_model_training_data(data_path, sizes, policy_Losses, value_Losses)

        print("SAVED")
        print()

def save(policy_nn, value_nn, policy_path, value_path):
    torch.save(policy_nn, policy_path)
    torch.save(value_nn, value_path)

def load(policy_path, value_path):
    policy_nn = torch.load(policy_path)
    policy_nn.eval()

    value_nn = torch.load(value_path)
    value_nn.eval()

    return policy_nn, value_nn

def train(env, policy_nn, value_nn, num_episodes, x):
    policy_optim = optim.Adam(policy_nn.parameters(), lr = LEARNING_RATE)
    value_optim = optim.Adam(value_nn.parameters(), lr = LEARNING_RATE)

    episode_lengths, policy_losses, value_losses = [], [], []

    for episode in range(num_episodes):

        states = torch.zeros(size=(MAX_EPISODE_DURATION * FPS, INPUT_DIMENSION))
        actions = torch.zeros(MAX_EPISODE_DURATION * FPS, dtype = int)
        rewards = torch.zeros(MAX_EPISODE_DURATION * FPS)
        dones = torch.zeros(MAX_EPISODE_DURATION * FPS)
        old_log_probs = torch.zeros(MAX_EPISODE_DURATION * FPS)

        state = env.reset()

        n = 0
        while True:
            action_probs = policy_nn(state).detach()
            action = np.random.choice(len(action_probs[0]), p = action_probs.numpy()[0])
            log_prob = torch.log(action_probs[0, action])

            next_state, reward, done = env.step(action)

            states[n] = state
            actions[n] = action
            rewards[n] = reward
            dones[n] = done
            old_log_probs[n] = log_prob

            state = next_state
            n += 1

            if done or n == MAX_EPISODE_DURATION * FPS:
                break

            del action_probs, action, log_prob, next_state
        

        returns = torch.zeros(MAX_EPISODE_DURATION * FPS)
        advantages = torch.zeros(MAX_EPISODE_DURATION * FPS)
        running_return = 0
        
        for t in reversed(range(n)):
            running_return = rewards[t] + GAMMA * running_return * (1 - dones[t])
            returns[t] = running_return

            advantages[t] = running_return - value_nn(states[t]).detach().item()


        # Update the policy and value networks
        for start in range(0, n, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_states = states[start:end]
            batch_actions = actions[start:end]
            batch_returns = returns[start:end]
            batch_advantages = advantages[start:end]
            batch_old_log_probs = old_log_probs[start:end]

            # Get current action probabilities and value estimates
            new_action_probs = policy_nn(batch_states)
            new_log_probs = torch.log(new_action_probs.gather(1, batch_actions.unsqueeze(1)).squeeze())

            ratio = torch.exp(new_log_probs - batch_old_log_probs)

            # Calculate surrogate loss
            surrogate_loss = ratio * batch_advantages
            clipped_surrogate_loss = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * batch_advantages

            policy_loss = -torch.min(surrogate_loss, clipped_surrogate_loss).mean()
            value_loss = (value_nn(batch_states) - batch_returns).pow(2).mean()

            # Update policy network
            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            # Update value network
            value_optim.zero_grad()
            value_loss.backward()
            value_optim.step()

            del end, batch_states, batch_actions, batch_returns, batch_advantages, batch_old_log_probs
            del new_action_probs, new_log_probs, ratio, surrogate_loss, clipped_surrogate_loss

        print(f"{x} - {episode + 1}/{num_episodes} - {n} - Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

        episode_lengths.append(n)
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        del states, state, actions, rewards, dones, old_log_probs
        del returns, advantages, running_return, n
        del policy_loss, value_loss
        gc.collect()
    
    return episode_lengths, policy_losses, value_losses  

def play(env, policy_nn):
    env = Environment()
    pygame.init()
    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    timer = pygame.time.Clock()

    iteration = 0
    run = True
    state = env.reset()
    while run:
        timer.tick(FPS)
        screen.fill(BACKGROUND_COLOR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        action_probs = policy_nn(state).detach()
        action = np.random.choice(len(action_probs[0]), p = action_probs.numpy()[0])

        next_state, reward, done = env.step(action)

        state = next_state
        iteration += 1

        if done or iteration == MAX_EPISODE_DURATION * FPS:
            iteration = 0
            state = env.reset()

        env.draw(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
