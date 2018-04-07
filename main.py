import json
import numpy as np

from Replay import ExperienceReplay
from Catch import Catch
from KerasWrapper import build_q_network


good_hyperparams =   {   
    "num_actions" : 3,
    "hidden_size" : 100,
    "batch_size" : 50,
    "grid_size" : 10,
    "learning_rate" : .2
}

model = build_q_network(good_hyperparams)

# Define environment/game
env = Catch(grid_size)
max_memory = 500
replayer = ExperienceReplay(max_memory=max_memory)

# Train

def explore_exploit_step(model, state, epsilon=.1) :
    if np.random.rand() <= epsilon :
        num_actions
        action = np.random.randint(0, , size=1)
    else:
        q = model.predict(state)
        action = np.argmax(q[0])

    return action

def train() :
    win_cnt = 0
    epochs = 1000
    explore = .1


    for e in range(epochs):
        loss = 0.
        env.reset()
        game_over = False
        state = env.observe()

        while not game_over:
            last_state = state
            action = explore_exploit_step(model)
            

            # apply action, get rewards and state
            state, reward, game_over = env.act(action)
            if reward == 1:
                win_cnt += 1

            # store experience (SARS step)
            experience = [last_state, action, reward, state]
            replayer.store_sars(experience, game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

            loss += model.train_on_batch(inputs, targets)

        print("Epoch {:03d}/999 | Loss {:.4f}"\
                + "| Win count {}".format(e, loss, win_cnt))


def save(model) :
    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)