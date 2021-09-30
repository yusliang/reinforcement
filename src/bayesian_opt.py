from bayes_opt import BayesianOptimization
from dqn1 import train_and_evaluate as dqn_train_and_evaluate
from policy_gradient import train_and_evaluate as pg_train_and_evaluate
from functools import partial
import gym

# buffer_size=100, steps=5000, batch_size=5, gamma=0.99, target_update_iter=400,
#                        learning_rate=0.0015, epsilon=0.1, epsilon_decay=0.995, min_epsilon=0.01,
#                        start_learning_steps=100
def search_dqn():
    pbounds = {'gamma': (0.9, 1), 'learning_rate': (0.001, 0.1), 'epsilon': (0.1, 0.2),
               'epsilon_decay': (0.99, 0.999), 'min_epsilon': (0, 0.01)}

    env = gym.make("CartPole-v0")
    partial_func = partial(dqn_train_and_evaluate, env=env, buffer_size=100, steps=5000, batch_size=5,
                           target_update_iter=400, start_learning_steps=100)

    optimizer = BayesianOptimization(
        f=partial_func,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=10,
    )


def search_pg():
    pbounds = {'steps': (100, 10000), 'gamma': (0.9, 1), 'learning_rate': (0.001, 0.1), 'epsilon': (0.1, 0.2),
               'epsilon_decay': (0.99, 0.999), 'min_epsilon': (0, 0.01), 'done_punish':(-100, 100)}
    env = gym.make("CartPole-v0")
    partial_func = partial(pg_train_and_evaluate, env=env)
    optimizer = BayesianOptimization(
        f=partial_func,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=10,
        n_iter=10,
    )


if '__main__' == __name__:
    search_pg()
