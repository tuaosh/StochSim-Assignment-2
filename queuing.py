import numpy as np

import itertools
import random
import simpy

seed = 42
numberOfServers = 2
serviceTime = 5
arrivalRate = 7
simTime = 20

# TODO: Rework serviceTime into service rate with Markov distribution

class Queue:
    def __init__(self, env: simpy.Environment, numberOfServers: int, serviceTime: float):
        self.env = env
        self.servers = simpy.Resource(env, numberOfServers)
        self.serviceTime = serviceTime

    def serve(self, customer):
        yield self.env.timeout(self.serviceTime)


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


def setup(env: simpy.Environment, num_machines: int, serviceTime: float, t_inter: int):
    queue = Queue(env, num_machines, serviceTime)

    customer_count = itertools.count()
    for _ in range(4):
        env.process(customer(env, f'Customer {next(customer_count)}', queue))

    while True:
        yield env.timeout(random.randint(t_inter - 2, t_inter + 2))
        env.process(customer(env, f'Customer {next(customer_count)}', queue))

if __name__ == "__main__":
    # Setup and start the simulation
    random.seed(seed)

    # Create an environment and start the setup process
    env = simpy.Environment()
    env.process(setup(env, numberOfServers, serviceTime, arrivalRate))

    # Execute!
    env.run(until=simTime)