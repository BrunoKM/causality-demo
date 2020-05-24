import pytest
import numpy as np
from causalitydemos.datasets import lotka_volterra


# TODO: Adjust tests post refractor
# @pytest.mark.parametrize(
#     'min_step, max_step, subsample_sim',
#     [
#         (1000, 3000, 5),
#         (1000, 1000, 10),
#         (1, 1001, 2)
#     ]
# )
# def test__variable_step_dataset__steps_within_range(min_step, max_step, subsample_sim):
#     simulator = lotka_volterra.LotkaVolterraSimulator(1.5, 2.5, 0.5, 0.6, time_delta=0.001)
#     dataset = lotka_volterra.VariableStepLotkaVolterraDataset(
#         simulator, size=1000, min_step_delta=min_step, max_step_delta=max_step,
#         samples_per_sim=100, subsample_sim=subsample_sim, seed=0, norm_step=False)
#     assert np.all(dataset.x[:, 2] <= max_step)
#     assert np.all(dataset.x[:, 2] >= min_step)


# @pytest.mark.parametrize(
#     'simulator_params, time_delta, min_step, max_step, subsample_sim, seed',
#     [
#         ([0.5, 0.5, 0.5, 0.6], 0.01, 1, 1000, 20, 0),
#         ([1.5, 2.5, 0.5, 0.6], 0.01, 1, 1000, 10, 1),
#         ([0.5, 0.4, 0.5, 0.6], 0.001, 1000, 1001, 20, 2),
#         ([2.5, 3.4, 2.5, 3.6], 0.009, 1000, 3000, 100, 4),
#     ]
# )
# def test__variable_step_dataset__produces_same_samples_as_sim(simulator_params, time_delta, min_step, max_step,
#                                                               subsample_sim, seed):
#     simulator = lotka_volterra.LotkaVolterraSimulator(*simulator_params, time_delta=time_delta)
#     dataset = lotka_volterra.VariableStepLotkaVolterraDataset(
#         simulator, size=1000, min_step_delta=min_step, max_step_delta=max_step,
#         samples_per_sim=50, subsample_sim=subsample_sim, seed=seed, norm_step=False)
#     assert len(dataset.x) == dataset.size
#     for i in range(len(dataset)):
#         x0 = dataset.x[i]
#         step_delta = int(x0[2])
#         sim_data = simulator.run_simulation(x0[0], x0[1], step_delta)
#         assert pytest.approx(dataset.y[i], rel=1e-7) == sim_data[step_delta]


# @pytest.mark.parametrize(
#     'simulator_params, min_time_delta, max_time_delta, samples_per_sim, seed',
#     [
#         ([0.5, 0.5, 0.5, 0.6], 0.01, 10, 10, 0),
#         ([0.1, 0.5, 0.3, 0.6], 0.1, 30, 30, 1),
#     ]
# )
# def test__variable_step_dataset_with_norm_step__produces_same_samples_as_sim(
#         simulator_params, min_time_delta, max_time_delta, samples_per_sim, seed):
#     simulator = lotka_volterra.LotkaVolterraSimulator(*simulator_params)
#     dataset = lotka_volterra.VariableStepLotkaVolterraDataset(
#         simulator, size=200, min_time_delta=min_time_delta, max_time_delta=max_time_delta,
#         samples_per_sim=samples_per_sim, seed=seed, norm_step=True)
#     for i in range(len(dataset)):
#         x0 = dataset.x[i]
#         step_delta = int(round((x0[2] * (max_step - min_step) + min_step)))
#         sim_data = simulator.run_simulation(x0[0], x0[1], step_delta)
#         assert pytest.approx(dataset.y[i], rel=1e-7) == sim_data[step_delta]

@pytest.mark.parametrize(
    'simulator_params, time_delta, samples_per_sim, seed',
    [
        ([0.5, 0.5, 0.5, 0.6],  0.3, 4, 0),
        ([1.5, 2.5, 0.5, 0.6],  5, 5, 1),
        ([0.5, 0.4, 0.5, 0.6],  1, 1, 2),
        ([2.5, 3.4, 2.5, 3.6], 10, 200, 4),
    ]
)
def test__fixed_step_dataset__produces_same_samples_as_sim(simulator_params, time_delta,
                                                           samples_per_sim, seed):
    simulator = lotka_volterra.LotkaVolterraSimulator(*simulator_params, rtol=1e-6)
    dataset = lotka_volterra.FixedStepLotkaVolterraDataset(
        simulator, size=500, time_delta=time_delta, samples_per_sim=samples_per_sim, seed=seed)
    assert len(dataset.x) == dataset.size
    assert len(dataset.y) == dataset.size
    for i in range(len(dataset)):
        x0 = dataset.x[i]
        sim_data = simulator.run_simulation(x0[0], x0[1], [time_delta])
        assert pytest.approx(dataset.y[i], rel=1e-3, abs=1e-5) == sim_data[0]
