from sympy import symbols, diff, Matrix, simplify

# Define variables
t, r, theta, phi = symbols('t r θ φ')
# Define metric tensor components g_{ij}
g_tt = -1
g_rr = 1 / (1 - r)
g_theta_theta = r**2
g_phi_phi = (r**2 * (sin(theta))**2)

# D
# 
# efine the metric tensor
g = Matrix([
    [g_tt, 0, 0, 0],
    [0, g_rr, 0, 0],
    [0, 0, g_theta_theta, 0],
    [0, 0, 0, g_phi_phi]
])

# Define the inverse metric tensor
g_inv = g.inv()

# Christoffel symbols calculation
def christoffel_symbols(g, g_inv, n):
    dim = len(g)
    christoffel = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                christoffel[i][j][k] = 0.5 * (
                    diff(g[i, k], n[j]) + 
                    diff(g[i, j], n[k]) - 
                    diff(g[j, k], n[i])
                )

    return christoffel

christoffel_symbols = christoffel_symbols(g, g_inv, [t, r, theta, phi])

# Riemann curvature tensor calculation
def riemann_tensor(christoffel, n):
    dim = len(christoffel)
    riemann = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    riemann[i][j][k] += christoffel[i][l][k] * christoffel[l][j][n] - christoffel[i][l][n] * christoffel[l][j][k]

    return riemann

riemann_tensor = riemann_tensor(christoffel_symbols, [t, r, theta, phi])

# Ricci tensor calculation
def ricci_tensor(riemann, n):
    dim = len(riemann)
    ricci = [[0 for _ in range(dim)] for _ in range(dim)]

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                ricci[i][j] += riemann[k][i][k][j]

    return ricci

ricci_tensor = ricci_tensor(riemann_tensor, [t, r, theta, phi])

# Print results
print("Christoffel Symbols:")
for i in range(len(christoffel_symbols)):
    for j in range(len(christoffel_symbols[i])):
        for k in range(len(christoffel_symbols[i][j])):
            print(f"Γ^{i}_{j}{k} =", simplify(christoffel_symbols[i][j][k]))

print("\nRiemann Curvature Tensor:")
for i in range(len(riemann_tensor)):
    for j in range(len(riemann_tensor[i])):
        for k in range(len(riemann_tensor[i][j])):
            for l in range(len(riemann_tensor[i][j][k])):
                print(f"R^{i}_{j}{k}{l} =", simplify(riemann_tensor[i][j][k][l]))

print("\nRicci Tensor:")
for i in range(len(ricci_tensor)):
    for j in range(len(ricci_tensor[i])):
        print(f"R_{i}{j} =", simplify(ricci_tensor[i][j]))
