
# projective_cubic_critical.sage
#
# Exact critical ideal for:
#   N=3, d=2, depth=3
#   f(z) = z + z^3, coordinatewise
#   normalize each sample after every activation
#   L = mean((Yhat-Y)^2)
#
# Run:
#   sage projective_cubic_critical.sage build
#   sage projective_cubic_critical.sage gb
#   sage projective_cubic_critical.sage decompose
#
# WARNING:
#   The full Groebner basis / decomposition may be extremely expensive.

import sys
from sage.all import *

MODE = sys.argv[1] if len(sys.argv) > 1 else "build"

# Exact coefficient field Q(sqrt(2)).
T = PolynomialRing(QQ, names=("w",))
w = T.gen()
K = NumberField(w^2 - 2, "s")
s = K.gen()

# Variables:
# 12 matrix parameters + 18 hidden-state coordinates
# + 9 normalization factors + 27 Lagrange multipliers = 66.
names = []

for ell in (1, 2, 3):
    for i in (1, 2):
        for j in (1, 2):
            names.append("a%d_%d%d" % (ell, i, j))

for ell in (1, 2, 3):
    for i in (1, 2, 3):
        for j in (1, 2):
            names.append("h%d_%d_%d" % (ell, i, j))

for ell in (1, 2, 3):
    for i in (1, 2, 3):
        names.append("r%d_%d" % (ell, i))

for j in range(27):
    names.append("mu%d" % j)

P = PolynomialRing(K, names=names, order="degrevlex")
v = P.gens_dict()

def A(ell):
    return matrix(P, 2, 2, [
        v["a%d_%d%d" % (ell, i, j)]
        for i in (1, 2) for j in (1, 2)
    ])

def h(ell, i):
    return vector(P, [
        v["h%d_%d_%d" % (ell, i, j)]
        for j in (1, 2)
    ])

def r(ell, i):
    return v["r%d_%d" % (ell, i)]

def F(z):
    return vector(P, [z[j] + z[j]^3 for j in range(2)])

A1, A2, A3 = A(1), A(2), A(3)

# Exact data. Each row is one sample.
X = [
    vector(P, [K(1), K(0)]),
    vector(P, [K(0), K(1)]),
    vector(P, [K(1)/s, K(1)/s])
]
Y = X

# 27 constraints:
# 18 propagation equations + 9 unit-norm equations.
constraints = []

for i in (1, 2, 3):
    # First activation from input X_i.
    constraints.extend([
        r(3, i)*h(3, i)[j] - F(A3*X[i-1])[j]
        for j in range(2)
    ])

    # Second activation.
    constraints.extend([
        r(2, i)*h(2, i)[j] - F(A2*h(3, i))[j]
        for j in range(2)
    ])

    # Final activation.
    constraints.extend([
        r(1, i)*h(1, i)[j] - F(A1*h(2, i))[j]
        for j in range(2)
    ])

    # Unit-norm constraints for all three normalized states.
    for ell in (1, 2, 3):
        constraints.append(
            h(ell, i).dot_product(h(ell, i)) - 1
        )

assert len(constraints) == 27

# L = mean squared error over all 3*2 scalar entries.
L = sum(
    (h(1, i)[j] - Y[i-1][j])^2
    for i in (1, 2, 3) for j in range(2)
) / 6

# All 39 non-multiplier variables.
parameter_vars = [
    v["a%d_%d%d" % (ell, i, j)]
    for ell in (1, 2, 3)
    for i in (1, 2) for j in (1, 2)
]

state_vars = [
    v["h%d_%d_%d" % (ell, i, j)]
    for ell in (1, 2, 3)
    for i in (1, 2, 3) for j in (1, 2)
]

normalizer_vars = [
    v["r%d_%d" % (ell, i)]
    for ell in (1, 2, 3) for i in (1, 2, 3)
]

zvars = parameter_vars + state_vars + normalizer_vars
mu = [v["mu%d" % j] for j in range(27)]

# Lagrange stationarity equations:
# grad(L) - sum_j mu_j grad(constraint_j) = 0.
stationarity = []

for z in zvars:
    stationarity.append(
        L.derivative(z)
        - sum(mu[j] * constraints[j].derivative(z)
              for j in range(27))
    )

equations = constraints + stationarity
Icrit = P.ideal(equations)

# The normalization factors must all be nonzero algebraically;
# the intended real network additionally requires every r > 0.
Rprod = prod(normalizer_vars)

print("Coefficient field:", K)
print("Number of ring variables:", P.ngens())
print("Number of constraints:", len(constraints))
print("Number of stationarity equations:", len(stationarity))
print("Number of generators in critical ideal:", len(equations))
print("Loss polynomial:", L)

if MODE == "build":
    save(Icrit, "critical_ideal.sobj")
    save((P, L, constraints, zvars, mu, Rprod),
         "critical_metadata.sobj")
    print("Saved critical_ideal.sobj and critical_metadata.sobj")

elif MODE == "gb":
    print("Computing Groebner basis of the full ideal...")
    print("This may require substantial memory and time.")
    G = Icrit.groebner_basis()
    save(G, "critical_groebner.sobj")
    print("Saved critical_groebner.sobj")
    print("Basis size:", len(G))

elif MODE == "decompose":
    print("Saturating by the product of normalization factors...")
    Isat = Icrit.saturation(P.ideal(Rprod))
    save(Isat, "critical_saturated.sobj")
    print("Saved critical_saturated.sobj")
    print("Starting primary decomposition...")
    dec = Isat.primary_decomposition()
    save(dec, "critical_primary_decomposition.sobj")
    print("Number of primary components:", len(dec))
    for k, Q in enumerate(dec):
        print("Component", k, "dimension:", Q.dimension())

else:
    raise ValueError("Use mode: build, gb, or decompose")
