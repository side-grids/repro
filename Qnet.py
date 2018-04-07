import numpy as np
import json

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from keras.utils import plot_model


# Grids only
class Qnet(object):
    
    def __init__(self, params:dict, weightsPath=None) :
        self.grid_size = params['grid_size']
        self.hidden_size = params['hidden_size']
        self.learning_rate = params['learning_rate']
        self.num_actions = params['num_actions']
        self.task = params['task']
        self.model = self.build_q_network(weightsPath)


    def build_q_network(self, weightsPath:str=None) -> Sequential :
        model = self._get_architecture_by_task_name()
        
        optimiser = sgd(lr=self.learning_rate)
        model.compile(optimiser, "mse")
        
        if weightsPath :
            model.load_weights(weightsPath)
        
        #plot_model(model, to_file='model.png')
        return model


    def _get_architecture_by_task_name(self) :
        nameToFunc = {
            "catch" : self.catch_qnet(),

        }
        return nameToFunc[self.task]



    def catch_qnet(self) :
        in_dim = self.grid_size**2
        model = Sequential()
        model.add( Dense(self.hidden_size, input_shape=(in_dim, ), 
                            activation='relu') )
        model.add( Dense(self.hidden_size, activation='relu') )
        model.add( Dense(self.num_actions) )     # Output layer

        return model


    def _get_layer_dims(self, layer_num:int) :
        layer = self.model.layers[layer_num]
        return layer.get_output_at(0).get_shape().as_list()


    def explore_exploit_step(self, state, epsilon=.1) :
        if np.random.rand() <= epsilon :
            action = self.get_random_action()
        else :
            action = self.get_optimal_action(state)

        return action


    def num_output_actions(self) :
        return self.model.output_shape[-1]


    def get_random_action(self) :
        num_actions = self.num_output_actions()
        return np.random.randint(0, num_actions, size=1)


    def get_optimal_action(self, state) :
        q = self.model.predict(state)
        return np.argmax(q[0])


    def save(self) :
        self.model.save_weights("models/model.h5", overwrite=True)
        with open("models/model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)


    def train(self, replayer, env, wins=0, epochs=1000, explore=.1, batch_size=50) :
        for e in range(epochs):
            loss = 0.
            env.reset()
            game_over = False
            state = env.observe()

            while not game_over:
                last_state = state
                action = self.explore_exploit_step(last_state)
                
                # apply action, get rewards and state
                state, reward, game_over = env.act(action)
                if reward == 1:
                    wins += 1

                # store experience (one SARS step)
                experience = [last_state, action, reward, state]
                replayer.store_sars(experience, game_over)

                # grab random bunch of memories, retrain
                states, targets = replayer.get_replay_batch(self.model, batch_size=batch_size)
                loss += self.model.train_on_batch(states, targets)

            print("Epoch {:03d}/{} | Loss {:.4f}".format(e, epochs, loss)\
                    + "| Win count {}".format(wins))


    def sars_step(self, env, state, returns) :
        action = self.get_optimal_action(state)
        state, reward, game_over = env.act(action)
        returns += reward
        
        return action, state, returns, game_over
        
            
    