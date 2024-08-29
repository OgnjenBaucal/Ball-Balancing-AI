import math
import random
from constants import *
import pygame
import torch

time_step = 1 / FPS
angle_step = MAX_ANGLE_CHANGE / FPS

class Environment:
    def __init__(self):
        sign = -1 if random.random() < 0.5 else 1
        self.x = sign * random.randint(50, 250)
        self.y = BALL_RADIUS
        self.x_speed = 0
        self.y_speed = 0
        self.teta = 0

    def reset(self):
        sign = -1 if random.random() < 0.5 else 1
        self.x = sign * random.randint(50, 250) 
        self.y = BALL_RADIUS
        self.x_speed = 0
        self.y_speed = 0
        self.teta = 0

        teta, dist, xspeed, yspeed = Environment.getState(self)
        return torch.FloatTensor([teta, dist, xspeed, yspeed]).unsqueeze(0)

    def step(self, action):
        alpha = self.teta
        sin_old = math.sin(self.teta)
        cos_old = math.cos(self.teta)
        sin_new = sin_old
        cos_new = cos_old

        if action == 1 and alpha < MAX_ANGLE:
            alpha += angle_step
        elif action == 2 and alpha > -MAX_ANGLE:
            alpha -= angle_step
        
        if alpha != self.teta:
            sin_new = math.sin(alpha)
            cos_new = math.cos(alpha)

            
            dist = Environment.distance(self)
            x_new = dist * cos_new - BALL_RADIUS * sin_new
            y_new = dist * sin_new + BALL_RADIUS * cos_new

            self.x = x_new
            self.y = y_new
            self.teta = alpha

            
            tmp = cos_new * cos_old + sin_new * sin_old
            if cos_old != 0: 
                self.x_speed *= (cos_new / cos_old) * tmp
            if(sin_old != 0):
                self.y_speed *= (sin_new / sin_old) * tmp

        
        tmp = sin_new * G * time_step
        self.x_speed += cos_new * tmp
        self.y_speed += sin_new * tmp

        
        self.x -= self.x_speed * time_step
        self.y -= self.y_speed * time_step

        teta, dist, xspeed, yspeed = Environment.getState(self)
        done = False
        reward = 0

        if NORMALIZED:
            done = (dist < -1) or (dist > 1)
            reward = Environment.reward(self, abs(dist), done)
        else:
            done = (dist < -(PLATFROM_LENGTH/2)) or (dist > (PLATFROM_LENGTH/2))
            reward = Environment.reward(self, abs(dist) / (PLATFROM_LENGTH/2), done)

        return torch.FloatTensor([teta, dist, xspeed, yspeed]).unsqueeze(0), reward, done
    
    def getState(self):
        dist = Environment.distance(self)
        if not NORMALIZED:
            return self.teta, dist, self.x_speed, self.y_speed
        
        alpha = self.teta / MAX_ANGLE
        dist /= (PLATFROM_LENGTH / 2)
        xspeed = self.x_speed / MAX_X_SPEED
        yspeed = self.y_speed / MAX_Y_SPEED
        return alpha, dist, xspeed, yspeed

    def reward(self, dist, done):
        dist_reward = (1 - dist) * DISTANCE_REWARD * time_step
        time_reward = TIME_REWARD * time_step
        total = 0
        
        if done:
            total = dist_reward + time_reward + FALL_PUNISHMENT
        else:
            total = dist_reward + time_reward 
        
        if NORMALIZED:
            return total
        else:
            return total * MAX_REWARD
          
    def distance(self):
        x1 = -BALL_RADIUS * math.sin(self.teta)
        y1 = BALL_RADIUS * math.cos(self.teta)
        result = math.sqrt((x1 - self.x)*(x1 - self.x) + (y1 - self.y)*(y1 - self.y))
        if x1 <= self.x:
            return result
        else:
            return -result

    def draw(self, screen):
        pygame.draw.circle(screen, BALL_COLOR, (X_V + self.x, Y_V - self.y - PLATFORM_THICKNES/2), BALL_RADIUS)

        x = (PLATFROM_LENGTH / 2) * math.cos(self.teta)
        y = (PLATFROM_LENGTH / 2) * math.sin(self.teta)
        pygame.draw.line(screen, PLATFROM_COLOR, (X_V - x, Y_V + y), (X_V + x, Y_V - y), PLATFORM_THICKNES)    
