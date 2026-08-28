


# ---- data
n_values = np.array([1, 10, 100, 500])
N_max = n_values.max()
Ns     = jnp.arange(1, N_max + 1)

pos = 6
left = slice(None, pos)
face = slice(pos, pos+3)
right = slice(pos + 3, None)

rates_nu = get_rates(graph, J, h, betas_jax, gammas_nu)
rates = rates_nu.sum(0)
rows, dE_J, dE_h_sign = graph
dE = J * dE_J + h * dE_h_sign
g = heat_rate_vectors(rates_nu, dE)
K = dense_generator_from_rates(graph[0], rates)

# ── Propagator and time evolution ─────────────
propagator = make_propagator(K)

G  = jnp.array(propagator(dt))

grid = sample_simplex(levels=60)
grid_eval = regularize_simplex(grid)
n_pts = grid_eval.shape[0]
simplex_face = jnp.zeros((n_pts, n_states)).at[:, face].set(grid_eval)

grid_xy = simplex_to_cartesian(grid)
triang = mtri.Triangulation(grid_xy[:, 0], grid_xy[:, 1])

pN, pts = get_traj(G, p0, N_max)
# shape (N_max, n_states)
traj = jnp.cumsum(pts[:-1], axis=0) / Ns[:, None]                  # broadcasting
q_stars = traj[n_values - 1]

traj3 = traj[:, face]
traj3 = traj3 / traj3.sum(axis=1, keepdims=True)
traj_xy = simplex_to_cartesian(traj3)

get_mmc = jax.vmap(lambda q: jnp.cumsum(KL(pts[:-1], q, axis=1) - KL(pts[1:], G @ q, axis=1)))

alpha = 0.95
tail_idx = jnp.concatenate([
  jnp.arange(pos),
  jnp.arange(pos + 3, n_states),
])

q_ref = q_stars[-1]
tail = jnp.clip(q_ref[tail_idx], 1e-8, None)
tail = tail / tail.sum()

simplex_face = jnp.zeros((n_pts, n_states))
simplex_face = simplex_face.at[:, face].set(alpha * grid_eval)
simplex_face = simplex_face.at[:, tail_idx].set((1 - alpha) * tail[None, :])


all_mmcs = get_mmc(simplex_face)
print(all_mmcs.shape)
mmcs = all_mmcs[:, n_values - 1]
print(mmcs.shape)


vmin= mmcs.min()
vmax = mmcs.max()
norm = mcolors.SymLogNorm(linthresh=1e-2, vmin=vmin, vmax=vmax)

# ---- figure
fig = plt.figure(figsize=(8.5,7.3), constrained_layout=True, dpi=300)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.06])

axs = [
  fig.add_subplot(gs[0, 0]),
  fig.add_subplot(gs[0, 1]),
  fig.add_subplot(gs[1, 0]),
  fig.add_subplot(gs[1, 1]),
]
cax = fig.add_subplot(gs[:, 2])

# fig, axes = plt.subplots(len(n_values) // 2, 2, figsize=(12, len(n_values) // 2 * 6), constrained_layout=True)
# axs: list[Axes] = onp.array(axes).flatten().tolist()

panel_labels = ["(a)", "(b)", "(c)", "(d)"]


mappable = None
for i, (ax, n) in enumerate(zip(axs, n_values)):
  values = mmcs[:, i]
  q_star = q_stars[i]
  q_star_simplex = normalize_simplex_point(q_star[face])
  q_star_xy = simplex_to_cartesian(q_star_simplex)

  # mappable = ax.tripcolor(triang, values, shading="gouraud", cmap="viridis", norm=norm, rasterized=True)
  # draw_simplex_frame(ax, color="0.20", lw=1.0)
  # add_corner_labels(ax, fontsize=8.5, color="0.25")
  # trajectory up to N
  # ax.plot(traj_xy[:n, 0], traj_xy[:n, 1], color="white", lw=1.2, alpha=0.85, zorder=3)
  # ax.scatter(*q_star_xy[:2], color="white", s=28, zorder=4)
  # ax.annotate(
  #   rf"$q^* = {format_prob(q_star[face])}$",
  #   xy=q_star_xy + np.array([0.02, -0.1]),
  #   xytext=(6, 6), textcoords="offset points", fontsize=16, color="k",
  #   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.6, alpha=0.7,),
  #   zorder=5,
  # )
  # ax.text(0.02, 0.97, panel_labels[i], transform=ax.transAxes, ha="left", va="top", fontsize=16, color="0.15")
  # ax.set_title(rf"$N = {n}$", pad=4)
  # ax.set(
  #   aspect='equal',
  #   xlim=(-0.01, 1.01), 
  #   ylim=(-0.01, SIMPLEX_VERTICES[2, 1] + 0.01),
  # )
  # ax.axis("off")

  ax.plot(np.arange(q_star.shape[0]), q_star)



# cb = fig.colorbar(mappable, cax=cax)
# cb.set_label(r"$\mathcal{M}_N(q;G,p_0)$", rotation=90, labelpad=10) # fontsize=14
# cb.ax.tick_params(labelsize=16)
# fig.savefig('simplex_heatmap.pdf')
plt.show()