import numpy as np
import torch
import tqdm
import math
import scipy
import scipy.integrate
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
import abc

sns.set()


class LotkaVolterraSimulator(object):
    def __init__(self, alpha: float, beta: float, gamma: float, delta: float, rtol: float = 1e-3) -> None:
        """

        Args:
            alpha, beta, gamma, delta: parameters of the Lotka-Volterra diff. equation
                (https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)
            rtol: Relative tolerance for the ODE solver 
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        # ODE Solver parameters
        self.rtol = rtol

    def run_simulation(self, init_x: float, init_y: float, eval_times: np.ndarray) -> np.array:
        """
        Run a simulation from the initial state and return the state evaluated at the given times.
        Args:
            init_x: float representing x (PREY) value at step 0
            init_y: float representing y (PREDATOR) value at step 0
            eval_times: array of times to evaluate the simulation at

        Returns:
            Array of shape [len(eval_times), 2] with the values of the state at each of the times in eval_times
        """
        if isinstance(eval_times, np.ndarray):
            pass
        elif isinstance(eval_times, (list, tuple)):
            eval_times = np.array(eval_times, dtype=np.float64)
        else:
            raise TypeError('eval_times has to be a 1D numpy array, or a list/tuple of type that'
                            ' can be cast to float')
        assert eval_times.ndim == 1
        init_state = np.array((init_x, init_y))
        sol = scipy.integrate.solve_ivp(self.system_dynamics_func, t_span=(0, eval_times[-1]),
                                        y0=init_state, vectorized=True, t_eval=eval_times, rtol=self.rtol)
        return sol.y.T

    def run_simulation_with_fixed_int(self, init_x: float, init_y: float, n_steps: int, time_delta: float = 1.0) -> np.array:
        """
        Run a simulation from the initial state given for a fixed number of steps. The returned array includes the
        initial state.

        Args:
            init_x: float representing x (PREY) value at step 0
            init_y: float representing y (PREDATOR) value at step 0
            n_steps: How many steps to simulate for
            time_delta: The time between each discretised step of the simulation

        Returns:
            A [n_steps + 1, 2] shaped numpy array, with the values of x and y at each step of the simulation.
        """
        if n_steps <= 0:
            raise ValueError(f'n_steps must be a positive integer, but was {n_steps}')
        return self.run_simulation(init_x, init_y, eval_times=np.arange(0, n_steps + 1, dtype=np.float64) * time_delta)

    def system_dynamics_func(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        The system dynamics function F, such that the ODE satisfies:
            dx/dt = F(t, x)
        Where x is the (2-dimensional) state vector.

        The time argument t needs to be present in the header for compliance with the scipy ODE-solver API, although
        it is not used to determine dx/dt.
        Args:
            t: time
            state: state vector of shape [2] or a set of k states of shape [2, k].

        Returns:
            An array of derivatives dy/dt, either of shape [2] or [2, k] if k state values were passed.
        """
        x, y = state[0], state[1]
        dx = self.alpha * x - self.beta * x * y
        dy = self.delta * x * y - self.gamma * y
        return np.stack((dx, dy), axis=0)


class SequenceLotkaVolterraDataset(Dataset):
    """
    Generates a dataset of Lotka-volterra state values samples at fixed steps:
    [x(0), x(delta_t), x(2*delta_t), ...]
    """
    def __init__(self, simulator: LotkaVolterraSimulator, size: int = 10000,
                 seq_length: int = 300, time_step_size: float = 1.,
                 seed: int = 0) -> None:
        """
        Args:
            simulator: a Lotka-Volterra simulator
            size: the number of examples to generate, where each example is a sequence of size seq_length
            seq_length: the sequence length of each example
            time_step_size: the fixed time interval between each sample in the sequence
            seed: random seed
        """
        self.simulator = simulator
        self._size = size
        self.seed = seed
        self.seq_length = seq_length
        self.time_step_size = time_step_size
        self.t = torch.arange(0., seq_length) * time_step_size
        self.data = self.generate_dataset()

    @property
    def size(self) -> int:
        return self._size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx], self.t
        return sample

    def generate_dataset(self) -> torch.Tensor:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        sim_data = []
        # Calculate the number of samples per simulation
        steps_per_sim = self.seq_length
        for sim in range(self.size):
            sim_results = self.simulator.run_simulation_with_fixed_int(np.random.lognormal(), np.random.lognormal(),
                                                                       steps_per_sim - 1,
                                                                       self.time_step_size)
            sim_results = torch.from_numpy(sim_results).float()

        sim_data = torch.stack(sim_data, dim=0)

        assert sim_data.shape[0] == self.size
        assert sim_data.shape[1] == self.seq_length
        assert sim_data.shape[2] == 2

        return sim_data


class LotkaVolterraDataset(Dataset):
    """Abstract Base Class for LV-based datasets."""
    def __init__(self, simulator, size, transform=None):
        self.simulator = simulator
        self.size = size
        self.transform = transform
        self.x: np.ndarray
        self.y: np.ndarray

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.x[idx], self.y[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample


class VariableStepLotkaVolterraDataset(LotkaVolterraDataset):
    """
    LotkaVolterra Simulation Data dataset with a variable time-step. The dataset has the form:
        [(x, t) -> (y)]
    where x is the initial state, and y is the state after time t.
    """

    def __init__(self, simulator: LotkaVolterraSimulator, size: int = 100000, min_time_delta: float = 1,
                 max_time_delta: float = 3000, samples_per_sim=100, seed: int = 0,
                 transform=None, norm_step=False) -> None:
        """
        Dataset with inputs x=(predator_population(t), prey_population(t), step_delta) and outputs
        y=(predator_population(t + step_delta), prey_population(t+step_delta)), where step_delta is picked uniformly
        from interval [min_step_delta, max_step_delta].

        Args:
            simulator: A Lotka-Volterra simulator
            size: The number of examples to generate
            min_time_delta: Minimum allowable value of step_delta inclusive
            max_time_delta: Maximum value of step_delta inclusive
            seed: random seed
            transform:
            norm_step: Whether to normalise time-step t to [0, 1] range (todo: not recommended)
        """
        super(VariableStepLotkaVolterraDataset, self).__init__(simulator=simulator, size=size, transform=transform)
        self.min_time_delta = min_time_delta
        self.max_time_delta = max_time_delta
        self.samples_per_sim = samples_per_sim
        self.seed = seed
        self.norm_step = norm_step
        self.x, self.y = self.generate_dataset()

    def generate_dataset(self) -> Tuple[np.array, np.array]:
        np.random.seed(self.seed)
        sim_data_x = []
        sim_data_y = []
        # Calculate the number of samples per simulation
        num_sims = math.ceil(self.size / self.samples_per_sim)
        # steps_per_sim = (self.samples_per_sim - 1) + 1 + self.max_time_delta
        for sim in tqdm.tqdm(range(num_sims)):
            # Sample values of time interval between observation x and y from interval [min_step, max_step]
            step_deltas = np.random.uniform(self.min_time_delta, self.max_time_delta, [self.samples_per_sim])
            # Sample the initial starting point
            start_times = np.random.uniform(0., self.max_time_delta, [self.samples_per_sim - 1])
            start_times = np.concatenate(([0.], start_times))  # At least one sample starts at time 0.

            eval_times = np.concatenate((start_times, start_times+step_deltas))
            sort_idxs = np.argsort(eval_times)  # Eval. times must be in sorted order
            reverse_sort_idxs = np.argsort(sort_idxs)  # Indices that reverse the sort 
            sim_results = self.simulator.run_simulation(
                np.random.lognormal(), np.random.lognormal(), eval_times=eval_times[sort_idxs])[reverse_sort_idxs]

            new_samples_x = sim_results[:self.samples_per_sim]
            new_samples_y = sim_results[self.samples_per_sim:]
            if self.norm_step:
                # Normalise to [0, 1] range
                # Shift for the minimum to be at 0
                new_step_deltas = (step_deltas[:, None] - self.min_time_delta).astype(np.float32)
                if self.max_time_delta != self.min_time_delta:
                    # Scale the maximum to be at 1
                    new_step_deltas /= self.max_time_delta - self.min_time_delta
            else:
                new_step_deltas = step_deltas[:, None]

            sim_data_x.append(np.concatenate((new_samples_x, new_step_deltas), axis=1))
            sim_data_y.append(new_samples_y)
        sim_data_x, sim_data_y = map(lambda l: np.concatenate(l, axis=0)[:self.size], (sim_data_x, sim_data_y))
        return sim_data_x, sim_data_y


class FixedStepLotkaVolterraDataset(LotkaVolterraDataset):
    """LotkaVolterra Simulation Data dataset with a fixed time-step."""

    def __init__(self, simulator: LotkaVolterraSimulator, size: int = 100000, time_delta: int = 400.,
                 samples_per_sim: int = 1, seed: int = 0, transform=None) -> None:
        super().__init__(simulator=simulator, size=size, transform=transform)
        self.time_delta = time_delta
        self.samples_per_sim = samples_per_sim 
        self.seed = seed
        self.x, self.y = self.generate_dataset()

    def generate_dataset(self) -> Tuple[np.array, np.array]:
        """
        Dataset with inputs x=(predator_population(t), prey_population(t)) and outputs
        y=(predator_population(t + time_delta), prey_population(t+time_delta)), where time_delta
        is fixed.

        Returns:
            Tuple[np.array, np.array]: x, y
        """
        np.random.seed(self.seed)
        sim_data_x = []
        sim_data_y = []
        num_sims = math.ceil(self.size / self.samples_per_sim)
        for sim in tqdm.tqdm(range(num_sims)):
            x_times = np.linspace(0, self.time_delta, self.samples_per_sim, endpoint=False)
            y_times = x_times + self.time_delta
            sim_results = self.simulator.run_simulation(
                np.random.lognormal(), np.random.lognormal(),
                eval_times=np.concatenate((x_times, y_times)))
            sim_data_x.append(sim_results[:self.samples_per_sim])
            sim_data_y.append(sim_results[self.samples_per_sim:])

        sim_data_x, sim_data_y = map(
            lambda l: np.concatenate(l, axis=0)[:self.size], (sim_data_x, sim_data_y))
        return sim_data_x, sim_data_y


class VariableStepIterableLvDataset(Dataset):
    """todo: A Lotka-Volterra Dataset that generates examples as needed, rather than storing them in an array."""
    pass


class ToTensor(object):
    """Convert numpy arrays in sample to pytorch Tensors."""
    def __call__(self, sample: Tuple[np.array, np.array]) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(sample[0]).float(), torch.from_numpy(sample[1]).float()
