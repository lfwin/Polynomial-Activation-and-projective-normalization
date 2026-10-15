# SageMath
K.<s> = QuadraticField(2)

R = PolynomialRing(
    K,
    names=('a','b','c','d','r1','r2','r3'),
    order='degrevlex'
)
a,b,c,d,r1,r2,r3 = R.gens()

def g(t):
    return t*(1+t^2)

def gp(t):
    return 1+3*t^2

def h(t):
    return t*(t^2+2)

def hp(t):
    return 2+3*t^2

u = a+b
v = c+d

Q1 = vector(R, [-g(c)*gp(a), g(a)*gp(c)])
Q2 = vector(R, [-g(d)*gp(b), g(b)*gp(d)])
Q3 = vector(R, [-h(v)*hp(u), h(u)*hp(v)])

S1 = g(a)^2 + g(c)^2
S2 = g(b)^2 + g(d)^2
S3 = h(u)^2 + h(v)^2

E1 = g(c)
E2 = -g(b)
E3 = h(v)-h(u)

eqs = [
    r1^2-S1,
    r2^2-S2,
    r3^2-S3,
]

for j in range(2):
    eqs.append(sqrt(2)*E1*r3^3*Q1[j] + E3*r1^3*Q3[j])
for j in range(2):
    eqs.append(sqrt(2)*E2*r3^3*Q2[j] + E3*r2^3*Q3[j])

I = R.ideal(eqs)
B = E1*E2*E3*r1*r2*r3

# 饱和，排除退化分支
Isat, exponent = I.saturation(R.ideal(B))

print("dimension =", Isat.dimension())
print("Groebner basis:")
print(Isat.groebner_basis())