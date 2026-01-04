import numpy as np

# ----------------- full Jacobian -----------------
def J_full(t1, t2, t3, L2, L3):
    c1, s1 = np.cos(t1), np.sin(t1)
    c2, s2 = np.cos(t2), np.sin(t2)
    c23, s23 = np.cos(t2+t3), np.sin(t2+t3)
    J = np.zeros((6, 3))
    # øverste 3 rader: lineær hastighet
    J[0, :] = [-s1*(L2*c2 + L3*c23),  -c1*(L2*s2 + L3*s23),  -c1*L3*s23]
    J[1, :] = [ c1*(L2*c2 + L3*c23),  -s1*(L2*s2 + L3*s23),  -s1*L3*s23]
    J[2, :] = [0.0,                    (L2*c2 + L3*c23),       L3*c23]
    # nederste 3 rader: vinkelhastighet
    J[3, :] = [0.0,  s1,  s1]
    J[4, :] = [0.0, -c1, -c1]
    J[5, :] = [1.0,  0.0, 0.0]
    return J

# ----------------- D-matrise og kinetisk energi -----------------
def compute_D(q, L, m, I_diag):
    θ1, θ2, θ3 = q
    L1, L2, L3 = L
    m1, m2, m3 = m
    I1x, I1y, I1z = I_diag[0]
    I2x, I2y, I2z = I_diag[1]
    I3x, I3y, I3z = I_diag[2]
    I1 = np.diag([I1x, I1y, I1z])
    I2 = np.diag([I2x, I2y, I2z])
    I3 = np.diag([I3x, I3y, I3z])
    # rotasjonsmatriser
    R1 = np.eye(3)
    c2, s2 = np.cos(θ2), np.sin(θ2)
    R2 = np.array([[c2, -s2, 0],[s2, c2, 0],[0,0,1]])
    c23, s23 = np.cos(θ2+θ3), np.sin(θ2+θ3)
    R3 = np.array([[c23, -s23, 0],[s23, c23, 0],[0,0,1]])
    I1b = R1 @ I1 @ R1.T
    I2b = R2 @ I2 @ R2.T
    I3b = R3 @ I3 @ R3.T
    # COM-Jacobianer, massesenter i midten
    J1 = np.zeros((6,3)); J1[3:,0] = [0,0,1]
    J2 = J_full(θ1, θ2, 0,  0.5*L2, 0)
    J3 = J_full(θ1, θ2, θ3, L2,      0.5*L3)
    def split(J): return J[:3,:], J[3:,:]
    Jv1, Jw1 = split(J1)
    Jv2, Jw2 = split(J2)
    Jv3, Jw3 = split(J3)
    D = (m1 * Jv1.T@Jv1 + Jw1.T@I1b@Jw1 +
         m2 * Jv2.T@Jv2 + Jw2.T@I2b@Jw2 +
         m3 * Jv3.T@Jv3 + Jw3.T@I3b@Jw3)
    return D

def kinetic_energy(q, qd, L, m, I_diag):
    D = compute_D(q, L, m, I_diag)
    dθ = np.asarray(qd).reshape(3,1)
    return 0.5 * (dθ.T @ D @ dθ).item(), D

# ------------- Nye funksjoner for C og g -------------

def compute_C(q, qd, L, m, I_diag, eps=1e-6):
    n = 3
    C = np.zeros((n,n))
    for k in range(n):
        dq_k = np.zeros(n); dq_k[k] = eps
        D_plus  = compute_D(q + dq_k, L, m, I_diag)
        D_minus = compute_D(q - dq_k, L, m, I_diag)
        dD_dqk  = (D_plus - D_minus)/(2*eps)
        for i in range(n):
            for j in range(n):
                dq_j = np.zeros(n); dq_j[j] = eps
                dq_i = np.zeros(n); dq_i[i] = eps
                dD_ik = (compute_D(q + dq_j, L, m, I_diag)[i,k] - 
                         compute_D(q - dq_j, L, m, I_diag)[i,k])/(2*eps)
                dD_jk = (compute_D(q + dq_i, L, m, I_diag)[j,k] - 
                         compute_D(q - dq_i, L, m, I_diag)[j,k])/(2*eps)
                c_ijk = 0.5*(dD_dqk[i,j] + dD_ik - dD_jk)
                C[i,j] += c_ijk * qd[k]
    return C

def compute_g(q, L, m):
    q1, q2, q3 = q
    L1, L2, L3 = L
    m1, m2, m3 = m
    g=9.81
    h1 = L1 / 2.0
    h2 = L1 + 0.5 * L2 * np.sin(q2)
    h3 = L1 + L2 * np.sin(q2) + 0.5 * L3 * np.sin(q2 + q3)

    g1 = 0.0
    g2 = g * (m2 * 0.5 * L2 * np.cos(q2)
              + m3 * (L2 * np.cos(q2) + 0.5 * L3 * np.cos(q2 + q3)))
    g3 = g * (m3 * 0.5 * L3 * np.cos(q2 + q3))

    return np.array([g1, g2, g3]).reshape(3, 1)


def compute_tau(q, qd, ddq, L, m, I_diag):
    D   = compute_D(q, L, m, I_diag)
    C   = compute_C(q, qd, L, m, I_diag)
    g   = compute_g(q, L, m)
    return D @ np.reshape(ddq, (3,1)) + C @ np.reshape(qd, (3,1)) + g

# ----------------- test -----------------

q   = np.deg2rad([90, -30, 45])  # rad
qd  = [0.4, 0.3, 0.2]             # rad/s
ddq = [0.1, 0.2, 0.3]             # rad/s^2
L   = [0.1009, 0.2221, 0.1362]    # m
m   = [0.3833, 0.2724, 0.1406]    # kg
I   = [[1.2e-4,1.2e-4,1.0e-4],
       [0.9e-4,0.9e-4,0.8e-4],
       [0.5e-4,0.5e-4,0.4e-4]]

K, D = kinetic_energy(q, qd, L, m, I)
C     = compute_C(q, qd, L, m, I)
g_vec = compute_g(q, L, m)
tau   = compute_tau(q, qd, ddq, L, m, I)

print("Kinetic energy K =", K, "J")
print("\nD(q) =\n", D)
print("\nC(q,qd) =\n", C)
print("\ng(q) =\n", g_vec)
print("\nTau =\n", tau)
