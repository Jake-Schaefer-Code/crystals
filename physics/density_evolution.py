import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm




def plot_density_matrix_heatmap(rho, title="Density Matrix"):
	"""
	Plots a 2D heatmap of the real part of rho.
	"""
	fig, ax = plt.subplots()
	cax = ax.matshow(rho.real, cmap='viridis')
	fig.colorbar(cax)
	ax.set_title(title)
	ax.set_xlabel("Column Index")
	ax.set_ylabel("Row Index")
	plt.show()

def animate_density_matrices(rho_list, interval=10):
  fig, ax = plt.subplots()
  cax = ax.matshow(rho_list[0].real, cmap='Blues')
  fig.colorbar(cax)
  ax.set_title("Time Evolution of Density Matrix (Real Part)")

  def update(frame):
      cax.set_array(rho_list[frame].real)
      ax.set_xlabel(f"Time step: {frame}")
      return [cax]

  anim = FuncAnimation(fig, update, frames=len(rho_list), interval=interval, blit=False)
  plt.show()


def evolve_density_matrix(rho0, H, dt, steps):
  # time-evolution for one time-step
  U_dt = expm(-1j * H * dt)
  rho = rho0.copy()
  rhos = [rho]
  for _ in range(steps):

      # Perform the discrete update: rho -> U_dt * rho * U_dt^\dagger
      rho = U_dt @ rho @ U_dt.conj().T
      
      # Optionally, ensure numerical stability: re-Hermitize and renormalize:
      rho = 0.5 * (rho + rho.conj().T)  # enforce Hermiticity
      trace_rho = np.trace(rho)
      if trace_rho != 0:
          rho /= trace_rho              # keep trace == 1 if small numerical drift

      rhos.append(rho)
  
  return rhos




def main():
  N = 8
  # rho_example = np.random.rand(N,N) + 1j*np.random.rand(N,N)

  # # make Hermitian
  # rho_example = 0.5 * (rho_example + rho_example.conj().T)
  # # trace-1, approximate density matrix
  # rho_example = rho_example / np.trace(rho_example)        

  # plot_density_matrix_heatmap(rho_example, title="Example Density Matrix (Real Part)")


  # rho_0 = np.random.rand(N,N) + 1j*np.random.rand(N,N)
  # rho_0 = 0.5*(rho_0 + rho_0.conj().T)  # Hermitian
  # rho_0 /= np.trace(rho_0)
  # ntimes = 100
  # rho_list = []
  # for t_idx in range(ntimes):
  #     phase = np.exp(1j * 2 * np.pi * t_idx / ntimes)
  #     rho_t = rho_0 * phase
  #     rho_t = 0.5 * (rho_t + rho_t.conj().T)
  #     rho_t /= np.trace(rho_t)
  #     rho_list.append(rho_t)

  # animate_density_matrices(rho_list, interval=60)

  H_rand = np.random.rand(N, N) + 1j*np.random.rand(N,N)
  H = 0.5 * (H_rand + H_rand.conj().T)
  
  # 2) Create an initial density matrix rho0 (trace 1, positive semidefinite)
  rand_mat = np.random.rand(N,N) + 1j*np.random.rand(N,N)
  rho0 = 0.5 * (rand_mat + rand_mat.conj().T)
  rho0 /= np.trace(rho0)
  
  # 3) Evolve for steps=20 using dt=0.1
  dt = 0.1
  steps = 20
  rho_list = evolve_density_matrix(rho0, H, dt, steps)
  
  # 4) Now visualize or animate the results
  # Example: just print trace to confirm it's ~1
  for i, r in enumerate(rho_list):
      print(f"Step {i}, Tr(rho) = {np.trace(r).real:.4f}")
  
  fig, axes = plt.subplots(1, 2, figsize=(15, 10))
  axes[0].imshow(rho_list[0].real, cmap='Blues')
  axes[1].imshow(rho_list[-1].real, cmap='Blues')
  # fig.colorbar()
  # For a quick heatmap at final step:
  # plt.imshow(rho_list[-1].real, cmap='viridis')
  # plt.colorbar()
  # plt.title("rho at final time (Real part)")
  plt.show()



if __name__ == "__main__":
  main()