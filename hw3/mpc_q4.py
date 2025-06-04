import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

def do_mpc(
    x0: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    W: np.ndarray,
    N: int,
    rx: float,
    ru: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Solve the MPC problem starting at state `x0`."""
    n, m = Q.shape[0], R.shape[0]
    x_cvx = cvx.Variable((N + 1, n))
    u_cvx = cvx.Variable((N, m))
    cost = cvx.quad_form(x_cvx[-1], P)
    constraints = [cvx.quad_form(x_cvx[-1], W) <= 1,
                   x_cvx[0] == x0, 
                   cvx.norm_inf(x_cvx, axis=1) <= rx,
                   cvx.norm_inf(u_cvx, axis=1) <= ru]
    for k in range(N):
        cost += cvx.quad_form(x_cvx[k], Q) + cvx.quad_form(u_cvx[k], R)
        constraints.append(x_cvx[k+1] == A @ x_cvx[k] + B @ u_cvx[k])
    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve(cvx.CLARABEL)
    x = x_cvx.value
    u = u_cvx.value
    status = prob.status

    return x, u, status


def solve_sdp(A, rx, n):
    M = cvx.Variable((n, n), symmetric=True)
    obj = cvx.log_det(M)
    block = cvx.bmat([[M, A@M], [(A@M).T, M]])
    constraints = [block >> 0, rx**2*np.eye(n) - M >> 0]
    prob = cvx.Problem(cvx.Maximize(obj), constraints)
    prob.solve(cvx.CLARABEL)
    return M.value

def generate_ellipsoid_points(M, num_points=100):
    """Generate points on a 2-D ellipsoid.
    The ellipsoid is described by the equation
    `{ x | x.T @ inv(M) @ x <= 1 }`,
    where `inv(M)` denotes the inverse of the matrix argument `M`.
    The returned array has shape (num_points, 2).
    """
    L = np.linalg.cholesky(M)
    θ = np.linspace(0, 2*np.pi, num_points)
    u = np.column_stack([np.cos(θ), np.sin(θ)])
    x = u @ L.T
    return x

# Part (d)
n, m = 2, 1
A = np.array([[0.9, 0.6], [0.0, 0.8]])
B = np.array([[0.0], [1.0]])
Q = np.eye(n)
R = np.eye(m)
P = np.eye(n)
N = 4
T = 15
rx = 5.0
ru = 1.0

M_val = solve_sdp(A, rx, n)
W = np.linalg.inv(M_val)
print(W)

ellipse_Xt = generate_ellipsoid_points(M_val)
# ellipse_AXt = generate_ellipsoid_points(np.linalg.inv(A.T @ W @ A))
ellipse_AXt = ellipse_Xt @ A.T
ellipse_stateconst = generate_ellipsoid_points(rx**2 * np.eye(2))

# Plot the ellipses
fig, ax = plt.subplots(dpi=150, figsize=(8, 8))
ax.plot(ellipse_Xt[:, 0], ellipse_Xt[:, 1], label="Ellipsoid Xt")
ax.plot(ellipse_AXt[:, 0], ellipse_AXt[:, 1], 'k-', label="Ellipsoid AXt")
ax.plot(ellipse_stateconst[:, 0], ellipse_stateconst[:, 1], '--', label="State Constraint")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Ellipsoids Visualization")
ax.legend()
ax.axis("equal")
fig.savefig("mpc_q4_ellipses.png", bbox_inches="tight")
plt.show()


x0 = np.array([0, -4.5])
fig, ax = plt.subplots(1, 2, dpi=150, figsize=(10, 8))
x = np.copy(x0)
x_mpc = np.zeros((T, N + 1, n))
u_mpc = np.zeros((T, N, m))
for t in range(T):
    x_mpc[t], u_mpc[t], status = do_mpc(x, A, B, P, Q, R, W, N, rx, ru)
    if status == "infeasible":
        x_mpc = x_mpc[:t]
        u_mpc = u_mpc[:t]
        break
    x = A @ x + B @ u_mpc[t, 0, :]
    ax[0].plot(x_mpc[t, :, 0], x_mpc[t, :, 1], "--*", color="k")
ax[0].plot(x_mpc[:, 0, 0], x_mpc[:, 0, 1], "-o", label="Trajectory")
ax[1].plot(u_mpc[:, 0], "-o", label='Control input')
ax[1].axhline(-1, linestyle='--', c='g')
ax[1].axhline(1, linestyle='--', c='g', label='Control bounds')
ax[0].plot(ellipse_Xt[:, 0], ellipse_Xt[:, 1], label="Ellipsoid Xt")
ax[0].plot(ellipse_AXt[:, 0], ellipse_AXt[:, 1], 'k-', label="Ellipsoid AXt")
ax[0].plot(ellipse_stateconst[:, 0], ellipse_stateconst[:, 1], '--', label="State Constraint")
ax[0].axis("equal")
fig.suptitle("MPC result")
ax[0].set_title('Trajectory and state constraints')
ax[1].set_title('Control Inputs')
ax[0].set_xlabel(r"$x_{k,1}$")
ax[1].set_xlabel(r"$k$")
ax[0].set_ylabel(r"$x_{k,2}$")
ax[1].set_ylabel(r"$u_k$")
ax[0].legend()
ax[1].legend()

fig.savefig("mpc_p4_sim.png", bbox_inches="tight")
plt.show()
