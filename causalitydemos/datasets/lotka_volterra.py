import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
import abc

sns.set()


class LotkaVolterraSimulator(object):
    def __init__(self, alpha: float, beta: float, gamma: float, delta: float, time_delta=1) -> None:
        """

        Args:
            alpha, beta, gamma, delta: parameters of the Lotka-Volterra diff. equation
                (https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)
            time_delta: The time between each discretised step of the simulation
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.time_delta = time_delta
        self.x: float = 0.
        self.y: float = 0.
        self.step: int = 0

    @property
    def state(self) -> np.array:
        return np.array([self.x, self.y])

    def init_state(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.step = 0

    def update_state(self) -> None:
        dx = self.alpha * self.x - self.beta * self.x * self.y
        dy = self.delta * self.x * self.y - self.gamma * self.y
        self.x += dx * self.time_delta
        self.y += dy * self.time_delta
        self.step += 1

    def run_simulation(self, init_x: float, init_y: float, n_steps: int) -> np.array:
        """
        Run a simulation from the initial state given for a fixed number of steps. The returned array includes the
        initial state.

        Args:
            init_x: float representing x value at step 0
            init_y: float representing y value at step 0
            n_steps: How many steps to simulate for

        Returns:
            A [n_steps + 1, 2] shaped numpy array, with the values of x and y at each step of the simulation.
        """
        if n_steps <= 0:
            raise ValueError(f'n_steps must be a positive integer, but was {n_steps}')

        self.init_state(init_x, init_y)

        simulated_states = np.zeros([n_steps + 1, 2])
        simulated_states[0] = self.state
        for step in range(1, n_steps + 1):
            self.update_state()
            simulated_states[step] = self.state
        return simulated_states


class LotkaVolterraDataset(Dataset):
    """Abstract Base Class for LV-based datasets."""
    def __init__(self, simulator, size, transform=None):
        self.simulator = simulator
        self.size = size
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.x[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample


class SequenceLotkaVolterraDataset(Dataset):
    def __init__(self, simulator: LotkaVolterraSimulator, size: int = 10000,
                 time_batch_size: int = 300, subsample_sim: int = 1,
                 seed: int = 0) -> None:
        """

        Args:
            simulator: A Lotka-Volterra simulator
            size: The number of examples to generate
            seed:
            transform:
        """
        self.simulator = simulator
        self._size = size
        self.seed = seed
        self.time_batch_size= time_batch_size
        self.subsample_sim = subsample_sim
        self.t = torch.arange(0., time_batch_size * subsample_sim, subsample_sim)
        self.x, self.y = self.generate_dataset()

    @property
    def size(self) -> int:
        return self._size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.x[idx], self.y[idx], self.t
        return sample

    def generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        sim_data_x = []
        sim_data_y = []
        # Calculate the number of samples per simulation
        steps_per_sim = self.time_batch_size * self.subsample_sim
        for sim in range(self.size):
            sim_results = self.simulator.run_simulation(np.random.lognormal(), np.random.lognormal(),
                                                        steps_per_sim - 1)
            sim_results = torch.from_numpy(sim_results).float()

            new_samples_x = sim_results[0]
            new_samples_y = sim_results[self.t.long()]

            sim_data_x.append(new_samples_x)
            sim_data_y.append(new_samples_y)
        sim_data_x = torch.stack(sim_data_x, dim=0)
        sim_data_y = torch.stack(sim_data_y, dim=0)

        assert sim_data_x.shape[0] == self.size
        assert sim_data_x.shape[1] == 2
        assert sim_data_y.shape[0] == self.size
        assert sim_data_y.shape[1] == self.time_batch_size
        assert sim_data_y.shape[2] == 2

        return sim_data_x, sim_data_y


class VariableStepLotkaVolterraDataset(LotkaVolterraDataset):
    """LotkaVolterra Simulation Data dataset with a variable time-step."""

    def __init__(self, simulator: LotkaVolterraSimulator, size: int = 100000, min_step_delta: int = 1,
                 max_step_delta: int = 3000, samples_per_sim=100, subsample_sim: int = 1, seed: int = 0, transform=None,
                 norm_step=False) -> None:
        """
        Dataset with inputs x=(predator_population(t), prey_population(t), step_delta) and outputs
        y=(predator_population(t + step_delta), prey_population(t+step_delta)), where step_delta is picked uniformly
        from interval [min_step_delta, max_step_delta].

        Args:
            simulator: A Lotka-Volterra simulator
            size: The number of examples to generate
            min_step_delta: Minimum allowable value of step_delta inclusive
            max_step_delta: Maximum value of step_delta inclusive
            subsample_sim:
            seed:
            transform:
            norm_step:
        """
        super(VariableStepLotkaVolterraDataset, self).__init__(simulator=simulator, size=size, transform=transform)
        self.min_step_delta = min_step_delta
        self.max_step_delta = max_step_delta
        self.samples_per_sim = samples_per_sim
        self.subsample_sim = subsample_sim
        self.seed = seed
        self.norm_step = norm_step
        self.x, self.y = self.generate_dataset()

    def generate_dataset(self) -> Tuple[np.array, np.array]:
        np.random.seed(self.seed)
        sim_data_x = []
        sim_data_y = []
        # Calculate the number of samples per simulation
        num_sims = math.ceil(self.size / self.samples_per_sim)
        steps_per_sim = (self.samples_per_sim - 1) * self.subsample_sim + 1 + self.max_step_delta
        for sim in range(num_sims):
            sim_results = self.simulator.run_simulation(np.random.lognormal(), np.random.lognormal(),
                                                        steps_per_sim - 1)

            # Sample values of step (time) interval between observation x and y from interval [min_step, max_step]
            step_deltas = np.random.randint(self.min_step_delta, self.max_step_delta + 1, [self.samples_per_sim])
            # Sample step value of the observation x
            start_steps = np.arange(0, self.samples_per_sim) * self.subsample_sim

            new_samples_x = sim_results[start_steps]
            new_samples_y = sim_results[start_steps + step_deltas]
            if self.norm_step:
                # Normalise to [0, 1] range
                # Shift for the minimum to be at 0
                new_step_deltas = (step_deltas[:, None] - self.min_step_delta).astype(np.float32)
                if self.max_step_delta != self.min_step_delta:
                    # Scale the maximum to be at 1
                    new_step_deltas /= self.max_step_delta - self.min_step_delta
            else:
                new_step_deltas = step_deltas[:, None]

            sim_data_x.append(np.concatenate((new_samples_x, new_step_deltas), axis=1))
            sim_data_y.append(new_samples_y)
        return tuple(map(lambda l: np.concatenate(l, axis=0)[:self.size], (sim_data_x, sim_data_y)))


class FixedStepLotkaVolterraDataset(LotkaVolterraDataset):
    """LotkaVolterra Simulation Data dataset with a fixed time-step."""

    def __init__(self, simulator: LotkaVolterraSimulator, size: int = 100000, step_delta: int = 3000,
                 subsample_sim: int = 1, seed: int = 0, transform=None) -> None:
        assert step_delta >= 1
        super(FixedStepLotkaVolterraDataset, self).__init__(simulator=simulator, size=size, transform=transform)
        self.step_delta = step_delta
        self.subsample_sim = subsample_sim
        self.seed = seed
        self.x, self.y = self.generate_dataset()

    def generate_dataset(self) -> Tuple[np.array, np.array]:
        np.random.seed(self.seed)
        sim_data_x = []
        sim_data_y = []
        num_sims = math.ceil(self.size * self.subsample_sim / self.step_delta)
        for sim in range(num_sims):
            sim_results = self.simulator.run_simulation(np.random.lognormal(), np.random.lognormal(), self.step_delta*2 - 1)
            sim_data_x.append(sim_results[:self.step_delta:self.subsample_sim])
            sim_data_y.append(sim_results[self.step_delta::self.subsample_sim])
        return tuple(map(lambda l: np.concatenate(l, axis=0)[:self.size], (sim_data_x, sim_data_y)))


class VariableStepIterableLvDataset(Dataset):
    pass


class ToTensor(object):
    """Convert numpy arrays in sample to pytorch Tensors."""

    def __call__(self, sample: Tuple[np.array, np.array]) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(sample[0]).float(), torch.from_numpy(sample[1]).float()
