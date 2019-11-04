import random
from enum import IntEnum
import matplotlib.pyplot as plt
import time
import numpy as np
import math


class Action(IntEnum):
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4


class LearnerType(IntEnum):
    TD_STATIC = 0
    Q_STATIC = 1
    Q_DYNAMIC = 2


class GridWorldLearner:

    ACTIONS = [Action.STAY, Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]

    def __init__(self, learner_type, epsilon, learning_rate, discount_rate):

        self.learner_type = learner_type

        self.grid_world = None
        self.x = None
        self.y = None
        self.goal_x = None
        self.goal_y = None
        self.reset_world()

        self.value_table = None
        self.q_table = None
        self.reset_learning()

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

        self.fig = None

    def reset_world(self):
        self.grid_world = [[-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [None, None, None, -1, -1],
                           [-1, -1, -1, -1, -1],
                           [-1, 20, -1, -1, -1]]

        self.goal_x = 9
        self.goal_y = 1

        valid = False
        while not valid:
            self.x = random.randint(0, 9)
            self.y = random.randint(0, 4)

            if self.grid_world[self.x][self.y] is not None:
                valid = True

    def reset_learning(self):

        self.value_table = [[10 for i in range(5)] for j in range(10)]
        self.q_table = [[[10 for i in range(5)] for j in range(10)] for k in range(5)]

    def get_indices(self, start_x, start_y, action):

        if action == Action.STAY:
            return start_x, start_y
        elif action == Action.MOVE_LEFT:
            return start_x - 1, start_y
        elif action == Action.MOVE_RIGHT:
            return start_x + 1, start_y
        elif action == Action.MOVE_UP:
            return start_x, start_y + 1
        elif action == Action.MOVE_DOWN:
            return start_x, start_y - 1
        else:
            raise Exception("Failed to give proper action for new indices. Gave %d" % action)

    def get_values(self):
        values = dict()

        random.shuffle(self.ACTIONS)
        for action in self.ACTIONS:
            indices_x, indices_y = self.get_indices(self.x, self.y, action)
            if (indices_x in range(10) and indices_y in range(5) and
                    (indices_x != 7 or indices_y >= 3)):
                if self.learner_type == LearnerType.TD_STATIC:
                    values[(indices_x, indices_y, action)] = self.value_table[indices_x][indices_y]
                elif self.learner_type == LearnerType.Q_STATIC or self.learner_type == LearnerType.Q_DYNAMIC:
                    values[(indices_x, indices_y, action)] = self.q_table[action][self.x][self.y]

        return values

    def choose_movement(self, return_best=False):
        next_values = self.get_values()

        sorted_choices = sorted(next_values, key=next_values.get, reverse=True)

        if return_best:
            choice = sorted_choices[0]
        else:
            p = random.random()
            for i, option in enumerate(sorted_choices):
                if p > self.epsilon ** (i + 1) or i == len(sorted_choices) - 1:
                    choice = option
                    break

        return choice[0:2], choice[2]

    def take_action(self):
        indices, action = self.choose_movement()

        old_x, old_y = self.x, self.y
        self.x, self.y = indices

        reward = self.grid_world[self.x][self.y]

        if self.x == self.goal_x and self.y == self.goal_y:
            predicted_next_value = 0
        else:
            next_move_x, next_move_y = None, None
            get_max = None
            if self.learner_type == LearnerType.TD_STATIC:
                get_max = False
            elif self.learner_type == LearnerType.Q_STATIC or self.learner_type == LearnerType.Q_DYNAMIC:
                get_max = True
            else:
                raise Exception("Invalid learner type")

            (next_move_x, next_move_y), next_action = self.choose_movement(return_best=get_max)

            if self.learner_type == LearnerType.TD_STATIC:
                predicted_next_value = self.value_table[next_move_x][next_move_y]
            elif self.learner_type == LearnerType.Q_STATIC or self.learner_type == LearnerType.Q_DYNAMIC:
                predicted_next_value = self.q_table[next_action][self.x][self.y]

        if self.learner_type == LearnerType.TD_STATIC:
            self.value_table[self.x][self.y] += self.learning_rate * (reward + self.discount_rate * predicted_next_value - self.value_table[self.x][self.y])
        elif self.learner_type == LearnerType.Q_STATIC or self.learner_type == LearnerType.Q_DYNAMIC:
            self.q_table[action][old_x][old_y] += self.learning_rate * (reward + self.discount_rate * predicted_next_value - self.q_table[action][old_x][old_y])

    def move_goal(self):
        self.grid_world[self.goal_x][self.goal_y] = -1

        random.shuffle(self.ACTIONS)
        for action in self.ACTIONS:
            indices_x, indices_y = self.get_indices(self.goal_x, self.goal_y, action)
            if (indices_x in range(10) and indices_y in range(5) and
                    (indices_x != 7 or indices_y >= 3)):
                self.goal_x, self.goal_y = indices_x, indices_y
                break

        self.grid_world[self.goal_x][self.goal_y] = 20

    def learn(self, num_episodes, record_interval, visualize=False, visualization_epoch=100):

        iterations = [None for i in range(math.ceil(num_episodes/record_interval))]
        rewards = [None for i in range(math.ceil(num_episodes/record_interval))]
        i = 0
        while i < num_episodes:

            self.reset_world()

            reward = 0

            for j in range(20):
                self.take_action()

                reward += self.grid_world[self.x][self.y]

                if visualize and i % visualization_epoch == 0:
                    self.display_world()

                if self.x == self.goal_x and self.y == self.goal_y:
                    break

                if self.learner_type == LearnerType.Q_DYNAMIC:
                    self.move_goal()

            if self.learner_type == LearnerType.Q_DYNAMIC:
                self.reset_world()

            if visualize and i % visualization_epoch == 0:
                time.sleep(0.2)

            if i % record_interval == 0:
                iterations[i//record_interval] = i
                rewards[i//record_interval] = reward

            i += 1

        if visualize:
            plt.close(1)

        return iterations, rewards

    def display_world(self):

        plt.figure(1)

        plt.clf()

        plt.scatter(self.goal_x + 1, self.goal_y + 1, s=200)

        plt.scatter(self.x + 1, self.y + 1, s=50)

        plt.plot([8, 8], [0, 3])

        plt.xlim([0.5, 10.5])
        plt.ylim([0.5, 5.5])

        plt.pause(0.01)

    def display_values(self):

        if self.learner_type == LearnerType.TD_STATIC:
            plt.figure(2)

            self.value_table[7][0:3] = [np.min(self.value_table)] * 3

            plt.imshow(np.array(self.value_table).T[::-1])

            plt.show()
        elif self.learner_type == LearnerType.Q_STATIC or self.learner_type == LearnerType.Q_DYNAMIC:
            fig, axes = plt.subplots(2, 3)

            for i in range(5):
                min_value = np.min(self.q_table[i])

                title = "Stay"
                if i == Action.MOVE_LEFT:
                    self.q_table[i][0][:] = [min_value] * 5
                    self.q_table[i][8][0:3] = [min_value] * 3
                    title = "Move Left"
                elif i == Action.MOVE_RIGHT:
                    self.q_table[i][9][:] = [min_value] * 5
                    self.q_table[i][6][0:3] = [min_value] * 3
                    title = "Move Right"
                elif i == Action.MOVE_UP:
                    for j in range(10):
                        self.q_table[i][j][4] = min_value
                    title = "Move Up"
                elif i == Action.MOVE_DOWN:
                    for j in range(10):
                        self.q_table[i][j][0] = min_value
                    self.q_table[i][7][3] = min_value
                    title = "Move Down"

                self.q_table[i][7][0:3] = [min_value] * 3

                axes[i % 2, i % 3].imshow(np.array(self.q_table[i]).T[::-1])
                axes[i % 2, i % 3].set_title(title)

            fig.delaxes(axes[1, 2])
            plt.show()

        else:
            raise Exception("Invalid learner type")


def main():
    visualize = True
    visualization_interval = 25

    display_values = True
    num_tests = 1 if display_values or visualize else 100

    num_iterations = 500
    recording_interval = 5

    epsilon = 0.1
    learning_rate = 0.50
    discount_rate = 0.9
    rewards = [None for i in range(num_tests)]
    for learner_type in [LearnerType.TD_STATIC, LearnerType.Q_STATIC, LearnerType.Q_DYNAMIC]:
        learner = GridWorldLearner(learner_type=learner_type, epsilon=epsilon, learning_rate=learning_rate,
                                   discount_rate=discount_rate)

        for i in range(num_tests):
            learner.reset_learning()
            iterations, rewards[i] = learner.learn(num_iterations, record_interval=recording_interval,
                                                   visualize=visualize, visualization_epoch=visualization_interval)
            rewards[i] = np.array(rewards[i])

        sum_rewards = rewards[0]
        for i in range(1, num_tests):
            sum_rewards += rewards[i]

        average_rewards = sum_rewards / num_tests

        if display_values:
            learner.display_values()
        else:
            plt.plot(iterations, average_rewards)

    if not display_values:
        plt.xlabel("Number of training episodes")
        plt.ylabel("Reward")
        plt.legend(["State based TD", "State, Action based Q", "State, Action based Q with moving goal"])
        plt.grid()
        plt.xlim([0, num_iterations])
        plt.ylim([-20, 20])
        plt.show()


if __name__ == "__main__":

    main()
