import numpy as np

import itertools
import random
import simpy

SEED = 42
generator: np.random.Generator = None

N_SERVER = 2
LAMBDA = 1
MU = 1
MEAN_SERVICE_TIME = 1 / LAMBDA
MEAN_ARRIVAL_TIME = 1 / MU

simTime = 20



class Queue:
    def __init__(self, env: simpy.Environment, numberOfServers: int, serviceRate: float):
        self.env = env
        self.servers = simpy.Resource(env, numberOfServers)
        self.serviceRate = serviceRate

    def serve(self, customer):
        yield self.env.timeout(generator.exponential(self.serviceRate))


def customer(env: simpy.Environment, name: str, queue: Queue):
    tQueue = env.now
    print(f'{name} enters the queue at {tQueue:.2f}.')
    with queue.servers.request() as request:
        yield request

        tStart = env.now
        print(f'{name} starts processing at {tStart:.2f}.')
        print(f'{name} waiting time {tStart - tQueue}')
        yield env.process(queue.serve(name))

        print(f'{name} leaves the queue at {env.now:.2f}.')


def setup(env: simpy.Environment, num_machines: int, serviceRate: float, arrivalRate: int, finish: simpy.Event):
    queue = Queue(env, num_machines, serviceRate)

    customer_count = itertools.count()
    for _ in range(4):
        env.process(customer(env, f'Customer {next(customer_count)}', queue))

    while True:
        yield env.timeout(generator.exponential(arrivalRate))
        env.process(customer(env, f'Customer {next(customer_count)}', queue))

        if env.now >= 20:
            finish.succeed()
        
def instantieateRNG(seed=None):
    global generator
    generator = np.random.default_rng(seed)

  

instantieateRNG()

if __name__ == "__main__":
    # Setup and start the simulation
    instantieateRNG(SEED)

    # Create an environment and start the setup process
    env = simpy.Environment()
    outEvent = simpy.Event(env)
    env.process(setup(env, N_SERVER, MEAN_SERVICE_TIME, MEAN_ARRIVAL_TIME, outEvent))

    # Execute!
    env.run(until=outEvent)