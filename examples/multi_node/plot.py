import pathlib

import numpy as np
import numpy.typing as npt
import numpy.linalg as npl
import matplotlib.pyplot as plt

output_path = pathlib.Path.cwd().parent.parent / "outputs" / "arepc_honest"

honest_nodes = ["1", "2", "3", "4", "5"]
all_states: list[npt.NDArray[np.float64]] = []
for node_idx in honest_nodes:
    states = np.load(output_path / f"node_{node_idx}_states.npy")
    all_states.append(states)

all_states_array = np.array(all_states)  # Shape: (n_honest_nodes, n_iter, n_state)
avg_states = np.mean(all_states_array, axis=0)  # Shape: (n_iter, n_state)
diffs = all_states_array - avg_states  # Shape: (n_honest_nodes, n_iter, n_state)
rmse = np.sqrt(np.mean(np.sum(diffs**2, axis=2), axis=0))  # RMSE over iterations

x0_mean = avg_states[0, :]  # First dimension of the average state over iterations
drift = npl.norm(avg_states - x0_mean[None, :], axis=1)  # Drift over iterations

fig, ax = plt.subplots()

ax.semilogy(rmse, label="Consensus Error (RMSE)")
ax.semilogy(drift, label="Drift from Initial Mean")

ax.set_xlabel("Iteration")
ax.minorticks_on()
ax.grid(which="major", linestyle="-", linewidth=0.8)
ax.grid(which="minor", axis="y", linestyle="--", linewidth=0.5)
ax.legend()

fig.tight_layout()

figure_path = pathlib.Path.cwd().parent.parent / "figures" / "arepc_honest"
figure_path.mkdir(parents=True, exist_ok=True)
fig.savefig(figure_path / "arepc_honest_performance.png", dpi=300)
