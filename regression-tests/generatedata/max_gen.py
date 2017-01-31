import random

mu = 8.0
ma1 = 0.566
delta_n1 = 0
exo1 = 0.12
exo2 = 0.65
exo3 = -0.333

for i in range(500000):
    err = random.uniform(-1.0, 1.0)
    maTerms = err + ma1*delta_n1
    x1 = random.uniform(-1.0, 1.0)
    x2 = random.uniform(-1.0, 1.0)
    x3 = random.uniform(-1.0, 1.0)
    exoTerms = exo1*x1 + exo2*x2 + exo3*x3
    output = maTerms + exoTerms + mu
    row = [str(i), str(output), str(x1), str(x2), str(x3)]
    print(",".join(row))
    delta_n1 = err
