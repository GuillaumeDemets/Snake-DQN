import random
import numpy as np
import gymnasium as gym
import math


class SnakeEnv(gym.Env):
    def __init__(self):
        self.row = 10
        self.col = 10
        self.state = np.zeros((self.row, self.col))
        self.snake = [[5, 5]]  # initial position of the snake
        self.state[self.snake[0][0], self.snake[0][1]] = 1
        self.score = 0
        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        self.direction = 0
        self.food = []
        self.place_food(1)
        self.truncated = False  # Episode stopped by the step limit
        self.terminated = False  # Collision or a completely filled board
        self.info = {}
        self.steps = 0

        ### Gym setups ###
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3, self.row, self.col), dtype=np.float32
        )  # channels: head, body, food

        self.action_space = gym.spaces.Discrete(4)  # action space: 4 directions

    def place_food(self, num_food=1):
        empty_space = []
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == 0:
                    empty_space.append([i, j])

        for i in range(num_food):
            if not empty_space:
                self.terminated = True
                return
            else:
                food = random.choice(empty_space)
                self.food.append(food)
                self.state[food[0], food[1]] = 2

    def _observation(self):
        observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        head_x, head_y = self.snake[0]
        observation[0, head_x, head_y] = 1
        for x, y in self.snake[1:]:
            observation[1, x, y] = 1
        for x, y in self.food:
            observation[2, x, y] = 1
        return observation

    def step(self, action):
        reward = 0.0
        old_head = self.snake[0]

        ### check impossible movement ###
        if self.direction == 0 and action == 2:
            action = self.direction
        if self.direction == 1 and action == 3:
            action = self.direction
        if self.direction == 2 and action == 0:
            action = self.direction
        if self.direction == 3 and action == 1:
            action = self.direction

        ### get the new head psoition ###
        x, y = self.directions[action]
        new_head = [self.snake[0][0] + x, self.snake[0][1] + y]

        ### check collisions ###
        if (
            new_head[0] < 0
            or new_head[0] >= self.row
            or new_head[1] < 0
            or new_head[1] >= self.col
        ):  # check wall collision
            reward -= 20
            self.terminated = True
            return (
                self._observation(),
                reward,
                self.terminated,
                self.truncated,
                self.info,
            )

        growing = new_head in self.food
        # The tail moves away unless the snake eats on this step.
        occupied = self.snake if growing else self.snake[:-1]
        if new_head in occupied:  # check self-collision
            reward -= 20
            self.terminated = True
            return (
                self._observation(),
                reward,
                self.terminated,
                self.truncated,
                self.info,
            )

        self.steps += 1
        if self.steps > len(self.snake) * 750:
            self.truncated = True
            reward -= 10.0
            return (
                self._observation(),
                reward,
                self.terminated,
                self.truncated,
                self.info,
            )

        if growing:  # check food collision
            self.snake.insert(0, new_head)
            self.food.remove(new_head)
            self.score += 1
            reward = 10.0
            self.place_food(1)  # place new food
            self.steps = 0
        else:  # move forward
            self.snake.insert(0, new_head)  # add head at new position
            self.snake.pop()  # remove the last one

        ### calculate reward ###
        if self.food:
            old_distance = abs(old_head[0] - self.food[0][0]) + abs(
                old_head[1] - self.food[0][1]
            )
            new_distance = abs(new_head[0] - self.food[0][0]) + abs(
                new_head[1] - self.food[0][1]
            )
            if new_distance < old_distance:
                reward += 1.0
            else:
                reward -= 1.5

        ### update state ###
        self.state = np.zeros((self.row, self.col))  # reset the field
        for x, y in self.snake:
            self.state[x, y] = 1  # mark the snake's body
        for fx, fy in self.food:
            self.state[fx, fy] = 2  # mark the food's position

        self.direction = action  # update the direction

        return (
            self._observation(),
            reward,
            self.terminated,
            self.truncated,
            self.info,
        )  # terminated: collision or full board; truncated: step limit

    def render(self):
        print("Score : ", self.score)
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == 0:
                    print(".", end=" ")
                elif self.state[i][j] == 1:
                    print("S", end=" ")
                elif self.state[i][j] == 2:
                    print("F", end=" ")

    def reset(self):
        self.state = np.zeros((self.row, self.col))
        self.snake = [[5, 5]]
        self.state[self.snake[0][0], self.snake[0][1]] = 1
        self.direction = 0
        self.score = 0
        self.alive = True
        self.food = []
        self.place_food(1)
        self.truncated = False
        self.terminated = False
        self.info = {}
        self.steps = 0
        return self._observation(), self.info
