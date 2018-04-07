import numpy as np

# An Experience is a list [state, game_over]


class ExperienceReplay(object) :
    def __init__(self, capacity=100, discount=.9):
        self.capacity = capacity
        self.memory = list()
        self.discount = discount


    def store_sars(self, state, game_over) :
        self.memory.append([state, game_over])
        if len(self.memory) > self.capacity:
            del self.memory[0]


    def get_replay_batch(self, model, batch_size=10) :
        states = self.get_shaped_zeroes(batch_size)
        num_actions = model.output_shape[-1]
        targets = np.zeros((states.shape[0], num_actions))
        randos = enumerate( np.random.randint(0, len(self.memory), 
                                                size=states.shape[0]) )
        
        for i, idx in randos :
            state, action, reward, next_state = self._retrieve_state(idx)
            game_over = self.memory[idx][1]

            states[i:i+1] = state
            targets[i] = model.predict(state)[0]
            Q_sa = np.max(model.predict(next_state)[0])

            if game_over:  
                targets[i, action] = reward
            else:
                targets[i, action] = self.get_forward_q_value(reward, Q_sa)

        return states, targets


    def get_shaped_zeroes(self, batch_size) :
        env_dim = self.memory[0][0][0].shape[1]
        required_cap = min(len(self.memory), batch_size)

        return np.zeros( (required_cap, env_dim) )


    def get_forward_q_value(self, reward, Q_sa) :
        return reward + (self.discount * Q_sa)


    def _retrieve_state(self, index) :
        return self.memory[index][0]
