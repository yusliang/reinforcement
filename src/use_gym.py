import gym
import pyglet
import time
print(pyglet.version)


env = gym.make('CartPole-v0') # insert your favorite environment

def some_random_games_first():
    # Each of these is its own game.
    for episode in range(5):
        env.reset()
        # this is each frame, up to 200...but we wont make it that far.
        for t in range(200):
            # This will display the environment
            # Only display if you really want to see it.
            # Takes much longer to display it.
            env.render()

            # This will just create a sample action in any environment.
            # In this environment, the action can be 0 or 1, which is left or right
            action = env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, done, info = env.step(action)
            print("{}.{}: {}, {}, {}".format(episode, t, reward, info, observation))
            time.sleep(0.1)
            if done:
                break


some_random_games_first()
time.sleep(5)

