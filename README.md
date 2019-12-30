# ModifiedNewtonFixedPointIteration
![Result](ModifiedNewtonFixedPointIteration/result.png)

# LagrangePolynomial
Lagrange polynomial interpolation using zeros of Chebyshev series.
## Result:
![Result](LagrangePolynomial/result.png)

# NonlinearRelaxation
Based on [my paper](NonlinearRelaxation/paper.pdf).

# UniformApproximation
The best uniform approximation is the approximation in space 
![space](https://latex.codecogs.com/gif.latex?R%3DC%20%5Cleft%5B%20a%2Cb%20%5Cright%5D),
<!---\left \| f \right \|_{C\left [ a,b \right ]} = \max_{x\epsilon \left [ a,b \right ]} \left | f(x) \right |-->
where ![metric](https://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7C%20f%20%5Cright%20%5C%7C_%7BC%5Cleft%20%5B%20a%2Cb%20%5Cright%20%5D%7D%20%3D%20%5Cmax_%7Bx%5Cepsilon%20%5Cleft%20%5B%20a%2Cb%20%5Cright%20%5D%7D%20%5Cleft%20%7C%20f%28x%29%20%5Cright%20%7C) — uniform metric.

## Result:
![Result](UniformApproximation/result.png)

# SimpsonIntegration
<!---{\int \limits _{a}^{b}f(x)dx}\approx {\frac  {b-a}{6}}{\left(f(a)+4f\left({\frac  {a+b}{2}}\right)+f(b)\right)}-->
![formula](https://latex.codecogs.com/gif.latex?%7B%5Cint%20%5Climits%20_%7Ba%7D%5E%7Bb%7Df%28x%29dx%7D%5Capprox%20%7B%5Cfrac%20%7Bb-a%7D%7B6%7D%7D%7B%5Cleft%28f%28a%29+4f%5Cleft%28%7B%5Cfrac%20%7Ba+b%7D%7B2%7D%7D%5Cright%29+f%28b%29%5Cright%29%7D)
## Output
```
Simpson:
	256	1.7076842519339281	1.7083880026331157
True value:
	1.718281828459045
```