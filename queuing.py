import itertools
import random
import simpy

seed = 42
numberOfServers = 2
serviceTime = 5
arrivalRate = 7
simTime = 20


class Queue:
    def __init__(self, env, numberOfServers, serviceTime):
        self.env = env
        self.server = simpy.Resource(env, numberOfServers)
        self.serviceTime = serviceTime

    def serve(self, customer):
        yield self.env.timeout(self.serviceTime)
        job = random.randint(50, 99)
        print(f"Completed {job}% of {customer}'s request.")


def customer(env, name, queue):
    print(f'{name} arrives at the queue at {env.now:.2f}.')
    with queue.server.request() as request:
        yield request

        print(f'{name} enters the queue at {env.now:.2f}.')
        yield env.process(queue.serve(name))

        print(f'{name} leaves the queue at {env.now:.2f}.')


def setup(env, num_machines, washtime, t_inter):
    queue = Queue(env, num_machines, washtime)

    customer_count = itertools.count()
    for _ in range(4):
        env.process(customer(env, f'Customer {next(customer_count)}', queue))

    while True:
        yield env.timeout(random.randint(t_inter - 2, t_inter + 2))
        env.process(customer(env, f'Customer {next(customer_count)}', queue))


# Setup and start the simulation
random.seed(seed)

# Create an environment and start the setup process
env = simpy.Environment()
env.process(setup(env, numberOfServers, serviceTime, arrivalRate))

# Execute!
env.run(until=simTime)