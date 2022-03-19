def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    print(game_state['field'])
    return game_state['user_input']
