import requests


class Handler:
    def __init__(self, args):
        self.addr = args.kas_selector_address

    def get_args(self):
        """TODO: get args"""
        j = requests.get(f'{self.addr}/arguments').json()
        assert 'sampler_args' in j and 'train_args' in j and 'extra_args' in j
        return j['sampler_args'], j['train_args'], j['extra_args']

    def get_path(self):
        j = requests.get(f'{self.addr}/kernel').json()
        assert 'path' in j
        return j['path']

    def success(self, name, state, reward):
        assert name is not None
        print(f'Posting: {self.addr}/success?name={name}${state}${reward}')
        requests.post(f'{self.addr}/success?name={name}${state}${reward}')

    def failure(self, name, state):
        assert name is not None
        print(f'Posting: {self.addr}/failure?name={name}${state}')
        requests.post(f'{self.addr}/failure?name={name}${state}')
