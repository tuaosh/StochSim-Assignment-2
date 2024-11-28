import numpy as np

import itertools
import random
import simpy

from typing import Tuple, List, Literal, Any, Generator

generator: np.random.Generator = None
PRIORITIES = Literal["", "F", "T"] 

def experiment(nServers: int, arrivalRate: float, serviceTimeDist: Generator, minSamples: int = 100, maxSamples: int= 10000, 
               seed: int=None, targetSTD: float=None, prioType: PRIORITIES = "", verbose=True):
    """
    Returns average waiting time and Standard deviation (S / sqrt(n)) for the queueing problem.

    Args:
        nServers: Amount of servers n available in the problem.
        Arrival rate: \\lambda for the inter-arrrival time distribution
        serviceTimeDist: Service Time distribution Generator with the desired properties
        minSamples: Minimum amount of samples to take (relevant when using STD termination)
        maxSamples: Maximum amounmt of samples to take before attempting to terminate.
        seed: Seed for the RNG.
        priotype: Type of queue priority to use. See `Customer()` for details.
        verbose: Wether to print customer status to stdOut. 
    """
    instantieateRNG(seed)

    # Create an environment and start the setup process
    env = simpy.Environment()
    outEvent = simpy.Event(env)
    samples = []
    env.process(setup(env, nServers, arrivalRate, serviceTimeDist, outEvent, targetSTD, minSamples, maxSamples, samples, prioType, verbose))
    
    # Execute!
    env.run(until=outEvent)
    
    return samples, np.std(samples, ddof=1) / np.sqrt(len(samples))

class Queue:
    def __init__(self, env: simpy.Environment, numberOfServers: int):
        self.env = env
        self.servers = simpy.PriorityResource(env, numberOfServers)

    def serve(self, serviceTime: float):
        yield self.env.timeout(serviceTime)


def customer(env: simpy.Environment, name: str, queue: Queue, serviceTime: float, prioType: PRIORITIES = '', 
             samples:List=None, verbose=True):
    """
    Customer process for simpy Environment
    Args:
        env: Simpy Environment
        name: Unique Customer Name
        queue: Queue object to request service access from
        serviceTime: Service time for customer task
        prioType: Character literal; F-FIFO, no priority queueing; T-Time, queue priority based on shortest time.
        samples: Optional greater scope list with wait time samples.
        vebose:  Wether to print customer status to stdout.
    """
    tQueue = env.now
    if verbose: print(f'{name} enters the queue at {tQueue:.2f}.')
    
    prioType = prioType.upper()
    priority = None
    if prioType in ['', 'F']:
        priority = 0
    elif prioType == 'T':
        priority = serviceTime
    else:
        raise ValueError(f"Unknown priority type {prioType}")


    with queue.servers.request(priority) as request:
        yield request

        tStart = env.now
        tWait  = tStart - tQueue

        if samples is not None:
            samples.append(tWait)

        if verbose:
            print(f'{name} starts processing at {tStart:.2f}.')
            print(f'{name} waiting time {tWait}')
        
        yield env.process(queue.serve(serviceTime))
        if verbose: print(f'{name} leaves the queue at {env.now:.2f}.')

def setup(env: simpy.Environment, nServer: int, arrivalRate: float, serviceTimeDist: Generator, finish: simpy.Event, 
          targetSTD: float, minSamples: int, maxSamples: int, samples: List, prioType: PRIORITIES= "", verbose=True):
    """
    Args:
        env: Simpy Environment
        nServer: Amount of servers in the system
        arrivalRate: Rate \\lamba that customers arrive in the queue.
        serviceTimeDist: Generator object that provides the desired service time distribution.
        finish: simpy Event object that signifies termintion.
        targetSTD:   Target standard deviation for use in general where to stop algorithm (lecture 3, slide 10) 
        minSamples: Minimum amount of sample points to take.
        maxSamples: Maximum amount of sample points to take.
        samples: Optional greater scope list with wait time samples.
        prioType: Character literal; F-FIFO, no priority queueing; T-Time, queue priority based on shortest time.
        vebose:  Wether to print customer status to stdout. 
    """
    prevLenSamples = 0
    arrivalTimeDist = markovTimeDist(arrivalRate)

    queue = Queue(env, nServer)

    customer_count = itertools.count()

    while True:
        yield env.timeout(next(arrivalTimeDist))

        serviceTime = next(serviceTimeDist)
        env.process(customer(env, f'Customer {next(customer_count)}', queue, serviceTime, prioType, samples, verbose))

        if len(samples) > prevLenSamples: # Only take stats if a new amount of samples are available/ 
            if not prevLenSamples % 1000:
                print(f"Samples: {prevLenSamples}")
            prevLenSamples = len(samples)

            std = np.std(samples, ddof = 1)

           
            if prevLenSamples > minSamples and targetSTD is not None:
                 # Algorithm from Lecture 3 page 10.
                if std / np.sqrt(prevLenSamples) <= targetSTD:
                    finish.succeed()
            if prevLenSamples > maxSamples:
                	finish.succeed()


def instantieateRNG(seed=None):
    """
    Instantiates the global RNG with the given seed.
    """
    global generator
    generator = np.random.default_rng(seed)


def markovTimeDist(rate: float):
    """
    Markov Time distribution generator, implemented such that it it trivial to develop other time distributions
    for the same experimet.

    Args:
        rate: \\Lambda for the exponential distribution
    
    Returns: A sampled timespan.
    """
    while True:
        yield generator.exponential(1 / rate)


def deterministicTimeDist(tInter: float):
    """
    Implementation of the deterministic time distribution such that it is compatible with markovTimeDist.
    Args:
        tInter: Fixed time for the interaction.
    Returns: Always returns tInter.
    """
    while True:
        yield tInter


def longtailHyperexponentialDist(rateA: float, rateB: float, probA: float):
    """
    Implementation of the suggested hyperexponential longtail-distribution given in assignment 2.4
    
    Args:
        rateA: Rate for exponential distribution A.
        rateB: Rate for exponential distribution B.
        probA: Probability of using rate A. Rate B is implied to be 1 - probA.


    Returns: an exponential distribution with \\lambda_A = rateA with probability p_A = probA
             and an exponential distribution with \\lambda_B = rateB with probability p_B = 1 - probA
    """
    while True:
        roll = generator.random()

        if probA < roll:
            yield generator.exponential(1 / rateA)
        else:
            yield generator.exponential(1 / rateB)


def saveTxt(fname: str, listOfLists: List[List[float]]) -> bool:
    """
    Takes a List of lists with different arbitrary lengths and exports it as a numpy array TXT file
    with missing values as NaN.

    Args:
        fname: the filename/path for the file
        listOfLists: The list of lists of arbitrary lengths.
    """
    maxLen = max([len(l) for l in listOfLists])

    out = np.empty((len(listOfLists), maxLen))
    out[:,:] = np.nan
    for i, l in enumerate(listOfLists):
        out[i,:len(l)] = l[:]

    np.savetxt(fname, out)

def readTxt(fname: str) -> List[List[float]]:
    """
    Reads a TXT file saved by saveTxt back into a list of lists.

    Args:
        fname: The filename/path of the file to read.

    Returns:
        List of lists of floats, reconstructed from the file.
    """
    # Load the file into a NumPy array
    data = np.loadtxt(fname)
        
    # Reconstruct the list of lists, ignoring NaN values
    listOfLists = [row[~np.isnan(row)].tolist() for row in data]

    return listOfLists


if __name__ == "__main__":
    # Setup and start the simulation
    N_SERVER = 2
    RHO = 0.9
    MU = 1
    LAMBDA = RHO * N_SERVER * MU
    SEED = None

    minSamples = 100
    maxSamples = 100

    l, s = experiment(N_SERVER, LAMBDA, markovTimeDist(MU), minSamples, maxSamples, SEED, 0.01, "F", True)
    print(f"mean: {np.mean(l)}")
    print(f"std:  {s}")