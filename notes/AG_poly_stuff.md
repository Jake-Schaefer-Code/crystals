

### Newton's Identities

Give relations between two types of *symmetric polynomials*, namely between *power sums* and *elementary symmetric polynomials*

For $k\geq 1$, the $k$-th power sum, denoted $p_k(x_1,\ldots, x_n)$ of variables $x_1,\ldots, x_n$ is 
$$
p_k(x_1,\ldots, x_n):=\sum_{i=1}^n x_i^k.
$$
For $k\geq 0$, the $k$-th elementary symmetric polynomial, denoted $e_k(x_1,\ldots, x_n)$ is the sum over all products of $k$ distinct variables.


Newton's identities can be stated as
$$
ke_k(x_1\ldots x_n) = \sum_{i=1}^k (-1)^{i-1}e_{k-i}(x_1,\ldots, x_n) p_i(x_1,\ldots, x_n).
$$
The LHS becomes zero after the $n$-th identity.



In terms of the characteristic polynomial of a matrix $A$ with eigenvalues $x_i$, the sum of the $x_i^k$, which is the $k$-th power sum $p_k$ of the roots of the characteristic polynomial of $A$, is given by the trace:
$$
p_k = \mathrm{tr}(A^k)
$$
The Newton identities now relate the traces of the powers $A^k$ to the characteristic polynomial of $A$ from the expansion of a polynomial with roots $x_i$:
$$
\prod_{i=1}^n (x-x_i) = \sum_{k=0}^n (-1)^k e_k x^{n-k}
$$


### Fundamental theorem of symmetric polynomials


### Pfaffians
For a skew-symmetric matrix $A$,
$$
\mathrm{pf}^2(A) = \mathrm{det}(A)
$$