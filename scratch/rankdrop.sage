
# SageMath
# Exact algebra over Q(sqrt(2)); no floating-point arithmetic.

K.<s> = NumberField(x^2 - 2)

P.<a,b,c,d> = PolynomialRing(K, order='degrevlex')

def g(t):
    return t + t^3

def gp(t):
    return 1 + 3*t^2

def Q(x, y):
    return vector(P, [
        -gp(x)*g(y),
         gp(y)*g(x)
    ])

p = vector(P, [a, c])
q = vector(P, [b, d])
w = (p + q) / s

Q1 = Q(p[0], p[1])
Q2 = Q(q[0], q[1])
Q3 = Q(w[0], w[1])

def det2(u, v):
    return u[0]*v[1] - u[1]*v[0]

D12 = det2(Q1, Q2)
D13 = det2(Q1, Q3)

I_rankdrop = P.ideal([D12, D13])

print("Degrees:", D12.total_degree(), D13.total_degree())
print("Groebner basis of rank-drop ideal:")
for f in I_rankdrop.groebner_basis():
    print(f)

save(I_rankdrop, "rankdrop_ideal.sobj")




