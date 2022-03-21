
import numpy as np

game_state_file = lambda x: f'logs/game_state{x}.npy'

def setup(self):
    pass

def act(self, game_state: dict):
    # save game_state
    step = game_state['step']
    np.save(game_state_file(step), game_state)

    self.logger.info('Pick action according to pressed key')
    print(repr(game_state['field']))
    return game_state['user_input']
