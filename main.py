import copy
import math
import numpy as np
import functools
import sympy as sp

from sympy import Matrix

import pdb


#####################################
# Make sure params aren't malformed #
#####################################

# should probably be rewritten in the final version, especially since it uses recursion which isn't great for Python
# THIS METHOD IS NO LONGER USED IN THE MAIN CODE
# Int -> Bool
def is_squarefree(x):
    if x < 0:
        x = -x

    if x == 1:
        return True

    def trial(num, cur):
        if num % cur == 0:
            return False
        if num == 1:
            return True
        step = 1
        while num % (cur + step) != 0:
            step += 1
        follow = cur + step
        return trial(num / follow, follow)

    smallest_divisor = 2
    while x % smallest_divisor != 0:
        smallest_divisor += 1

    return trial(x / smallest_divisor, smallest_divisor)


# List(Int) -> Bool
def are_pairwise_coprime(l):
    return functools.reduce(lambda x, y: x and y, list(map(lambda t: np.gcd(t[0], t[1]) == 1, [[l[i], l[j]]
                                                                                               for i in range(len(l))
                                                                                               for j in range(len(l))
                                                                                               if i != j])))


# Int -> Int -> Int -> (Int, Int, Int)
def reduce(a, b, c):
    common_factor = np.gcd.reduce([a, b, c])
    return a // common_factor, b // common_factor, c // common_factor


# Int -> Int -> Int -> (Int, Int, Int)
def remove_squares(a, b, c):
    return tuple(map(lambda n: functools.reduce(lambda x, y: int(x * y), [factor for factor in sp.factorint(n)], 1),
                     [a, b, c]))


# Int -> Int -> Int -> Bool
def check_reduced_params(a, b, c, strict_squarefree=False):
    exception_string = ""
    if len(set([x < 0 for x in [a, b, c]])) < 2:
        exception_string += "\n - all have the same sign"
    if not are_pairwise_coprime([a, b, c]):
        exception_string += "\n - are not pairwise coprime."
    if strict_squarefree and not functools.reduce(lambda x, y: x and y, list(map(is_squarefree, [a, b, c]))):
        exception_string += "\n - are not squarefree."
    if exception_string != "":
        for n in range(2, 50):
            soln_list = [(x, y, z) for x in range(0, n) for y in range(0, n) for z in range(0, n)
                         if (a * x ** 2 + b * y ** 2 + c * z ** 2) % n == 0 if not (x == y == z == 0)]
            if not soln_list:
                exception_string += "The given form has no solutions mod", n
                break
        raise Exception("Reduced params:" + exception_string)
    if not is_quadratic_residue(-b * c, a):
        exception_string += "-bc is not a quadratic residue mod a.\n"
    if not is_quadratic_residue(-a * c, b):
        exception_string += "-ac is not a quadratic residue mod b.\n"
    if not is_quadratic_residue(-a * b, c):
        exception_string += "-ab is not a quadratic residue mod c.\n"
    if exception_string != "":
        for n in range(2, 50):
            soln_list = [(x, y, z) for x in range(0, n) for y in range(0, n) for z in range(0, n)
                         if (a * x ** 2 + b * y ** 2 + c * z ** 2) % n == 0 if not (x == y == z == 0)]
            if not soln_list:
                exception_string += "The given form has no solutions mod", n
                break
        raise Exception(exception_string)


################################
# Stuff for square roots, etc. #
################################

# Int -> Primes -> {-1, 1}
def legendre_symbol(a, p):
    if a == 0:
        raise Exception("This shouldn't happen, right?")
    if p == 2:
        return 1
    return 1 if pow(a, (p - 1) // 2, int(p)) == 1 else -1


# Vec(Int, a) -> Vec(Int, a) -> Bool
def is_quadratic_residue(a, n):
    prime_factors = sp.factorint(n).keys()
    if -1 in prime_factors:
        prime_factors = list(prime_factors)
        prime_factors.remove(-1)
    return not (-1 in [legendre_symbol(a, p) for p in prime_factors])


# Taken from https://en.wikibooks.org/wiki/Algorithm_Implementation/Mathematics/Extended_Euclidean_algorithm#Python
# Int -> Int -> (Int, Int, Int)
def xgcd(a, b):
    """return (g, x, y) such that a*x + b*y = g = gcd(a, b)"""
    x0, x1, y0, y1 = 0, 1, 1, 0
    while a != 0:
        q, b, a = b // a, a, b % a
        y0, y1 = y1, y0 - q * y1
        x0, x1 = x1, x0 - q * x1
    if b < 0:
        b *= -1
        x0 *= -1
        y0 *= -1
    return b, x0, y0


# Vec(Int, a) -> Vec(Int, a) -> Int
def chinese_remainder(a, n):
    n = list(map(lambda x: np.abs(x), n))

    if len(n) != len(a):
        raise Exception("The number of cosets differs from the number of moduli.")

    if len(n) == 1:
        return a[0]

    def crt2(a, n):
        if len(a) != 2 or len(n) != 2:
            raise Exception("The list of cosets and the list of moduli should have length 2.")

        _, m_1, m_2 = xgcd(n[0], n[1])

        return a[0] * m_2 * n[1] + a[1] * m_1 * n[0]

    while len(n) != 2:
        a_prime = crt2(a[0:2], n[0:2])
        n_prime = n[0] * n[1]

        a = [a_prime] + a[2:]
        n = [n_prime] + n[2:]

    return crt2(a, n)


# Implementation of Tonalli-Shanks
# Taken from https://eli.thegreenplace.net/2009/03/07/computing-modular-square-roots-in-python
# xrange changed to range for use with Python 3
# Int -> Primes -> Int
def modular_sqrt(a, p):
    """ Find a quadratic residue (mod p) of 'a'. p
        must be an odd prime.

        Solve the congruence of the form:
            x^2 = a (mod p)
        And returns x. Note that p - x is also a root.

        0 is returned is no square root exists for
        these a and p.

        The Tonelli-Shanks algorithm is used (except
        for some simple cases in which the solution
        is known from an identity). This algorithm
        runs in polynomial time (unless the
        generalized Riemann hypothesis is false).
    """
    # Simple cases
    #
    if legendre_symbol(a, p) != 1:
        return 0
    elif a == 0:
        return 0
    elif p == 2:
        return 0
    elif p % 4 == 3:
        return pow(a, (p + 1) / 4, p)

    # Partition p-1 to s * 2^e for an odd s (i.e.
    # reduce all the powers of 2 from p-1)
    #
    s = p - 1
    e = 0
    while s % 2 == 0:
        s /= 2
        e += 1

    # Find some 'n' with a legendre symbol n|p = -1.
    # Shouldn't take long.
    #
    n = 2
    while legendre_symbol(n, p) != -1:
        n += 1

    # Here be dragons!
    # Read the paper "Square roots from 1; 24, 51,
    # 10 to Dan Shanks" by Ezra Brown for more
    # information
    #

    # x is a guess of the square root that gets better
    # with each iteration.
    # b is the "fudge factor" - by how much we're off
    # with the guess. The invariant x^2 = ab (mod p)
    # is maintained throughout the loop.
    # g is used for successive powers of n to update
    # both a and b
    # r is the exponent - decreases with each update
    #
    x = pow(a, (s + 1) / 2, p)
    b = pow(a, s, p)
    g = pow(n, s, p)
    r = e

    while True:
        t = b
        m = 0
        for m in range(r):
            if t == 1:
                break
            t = pow(t, 2, p)

        if m == 0:
            return x

        gs = pow(g, 2 ** (r - m - 1), p)
        g = (gs * gs) % p
        x = (x * gs) % p
        b = (b * g) % p
        r = m


# Calculate sqrt(-bc) mod a
# Int -> Int -> Int
def calculate_root(b, c, a):
    if np.abs(a) == 1:
        return 0
    prime_factors = sp.factorint(a)
    roots = [sp.ntheory.residue_ntheory._sqrt_mod_prime_power(-b * c, p, prime_factors[p])[0]
             for p in prime_factors.keys() if p != -1]
    mods = [p ** prime_factors[p] for p in prime_factors.keys() if p != -1]
    return chinese_remainder(roots, mods)


################################
# Find big A, big B, and big C #
################################

# Int -> Int -> Int -> Int -> Int -> Int -> (Int, Int, Int)
def find_ABC(small_a, small_b, small_c, goth_a, goth_b, goth_c):
    A_residue = [small_c, goth_c]
    A_mods = [small_b, small_c]
    B_residue = [small_a, goth_a]
    B_mods = [small_c, small_a]
    C_residue = [small_b, goth_b]
    C_mods = [small_a, small_b]
    A = chinese_remainder(A_residue, A_mods)
    B = chinese_remainder(B_residue, B_mods)
    C = chinese_remainder(C_residue, C_mods)
    gcd_ABC = np.gcd.reduce([A, B, C])
    A = A // gcd_ABC
    B = B // gcd_ABC
    C = C // gcd_ABC
    return A, B, C


##########################
# EEA for three integers #
##########################

# Int -> Int -> Int -> (Int, Int, Int, Int, Int, Int, Int, Int, Int)
def get_g_transformation(aa, bb, cc):
    """Goal: find α',β',γ',α",β",γ", such that β'γ"-γ'β"=Aa, γ'α"-α'γ"=Bb, α'β"-β'α"=Cc given Aa, Bb, Cc"""
    bc_gcd, beta, gamma = xgcd(bb, cc)
    abc_gcd, alpha, factor = xgcd(aa, bc_gcd)
    beta *= factor
    gamma *= factor
    # Start to implement Lemma 279
    arb1 = 1
    arb2 = 2
    arb3 = 3
    if arb2 * gamma - arb3 * beta == 0 and arb3 * alpha - arb1 * gamma == 0 and arb1 * beta - arb2 * alpha == 0:
        arb3 += 1
    bl = arb2 * gamma - arb3 * beta
    bl1 = arb3 * alpha - arb1 * gamma
    bl2 = arb1 * beta - arb2 * alpha
    bl12_gcd, b1, b2 = xgcd(bl1, bl2)
    beta_gcd, b, factor = xgcd(bl, bl12_gcd)
    b1 *= factor
    b2 *= factor
    alpha2 = int((bb * bl2 - cc * bl1) / beta_gcd / abc_gcd)
    beta2 = int((cc * bl - aa * bl2) / beta_gcd / abc_gcd)
    gamma2 = int((aa * bl1 - bb * bl) / beta_gcd / abc_gcd)
    h = b * aa + b1 * bb + b2 * cc
    alpha1 = int(b - h * alpha)
    beta1 = int(b1 - h * beta)
    gamma1 = int(b2 - h * gamma)
    # return alpha, beta, gamma, alpha1, beta1, gamma1, alpha2, beta2, gamma2
    return Matrix([[alpha, alpha1, alpha2],
                   [beta, beta1, beta2],
                   [gamma, gamma1, gamma2]])


###########################
# Binary reduction theory #
###########################

# Implementation of Art 171.
# Note that positive determinant means that b^2 - ac will be negative,
# i.e. we use the modern convention for the determinant, *not* Gauss's convention.
# This is iterative and not recursive because Python
# Vec(Vec(Int, 2), 2) -> Vec(Vec(Int, 2), 2)
def positive_determinant_binary_reduction(form):
    determinant = int(form.det())

    if determinant <= 0:
        raise Exception("The determinant of the form", form, "isn't positive.")

    a = form[0, 0]
    b = form[0, 1]
    a_prime = form[1, 1]

    if np.sqrt(4. * determinant / 3) >= a >= 2 * b and a <= a_prime:
        raise Exception("Something's wrong. This method shouldn't have been called, right? Or is the condition wrong?")
        # return Matrix([[1, 0], [0, 1]])

    running_form = form_tuple = (a, b, a_prime)
    transformation = Matrix([[1, 0], [0, 1]])

    b_prime = 0

    # I stand by this
    while True:
        # temp_form = running_form

        # b_prime = min([m for m in range(-np.abs(form_tuple[2]) + 1, np.abs(form_tuple[2]))
        #                if (m % form_tuple[2] == (-form_tuple[1]) % form_tuple[2])], key=np.abs)

        b_prime = (-form_tuple[1]) % np.abs(form_tuple[2])
        if b_prime > np.abs(form_tuple[2]) / 2:
            b_prime -= np.abs(form_tuple[2])

        if form_tuple[2] % 2 == 0 and np.abs(b_prime) == np.abs((form_tuple[2] // 2)):
            b_prime = np.abs(b_prime)

        # Note that we can add the determinant for the third component, since Gauss uses the negation of his determinant
        running_form = (form_tuple[2], b_prime, (b_prime ** 2 + determinant) // form_tuple[2])

        # Art 160
        # Double check this
        transformation = transformation * Matrix([[0, -1], [1, (form_tuple[1] + running_form[1]) // form_tuple[2]]])

        form_tuple = running_form  # replaced temp form with running form

        if np.abs(running_form[2]) >= np.abs(running_form[0]):
            break

    return transformation


# Implementation of Art 183
# Vec(Vec(Int, 2), 2) -> Vec(Vec(Int, 2), 2)
def negative_nonsquare_determinant_binary_reduction(form):
    determinant = int(form.det())

    if determinant >= 0:
        raise Exception("The determinant of the form", form, "isn't negative.")

    # so we can use Gauss's convention
    determinant *= -1

    if functools.reduce(lambda x, y: x and y, list(map(lambda x: x % 2 == 0, sp.factorint(determinant))), True):
        raise Exception("The negated determinant of the form", form, "is a perfect square.")

    a = form[0, 0]
    b = form[0, 1]
    a_prime = form[1, 1]

    # pdb.set_trace()

    # ceil is used for the right hand limit since range is exclusive on the right but inclusive on the left
    if 0 < b < np.sqrt(determinant) and np.abs(a) in range(math.ceil(np.sqrt(determinant) - b),
                                                           math.ceil(np.sqrt(determinant) + b)):
        raise Exception("Something's wrong. This method shouldn't have been called, right?")

    running_form = form_tuple = (a, b, a_prime)
    transformation = Matrix([[1, 0], [0, 1]])

    # I stand by this
    while True:
        # temp_form = running_form

        b_prime = (-form_tuple[1]) % form_tuple[2]

        # the float cast is there because it makes the code work. I don't really care why.
        while b_prime not in range(int(np.ceil(float(np.sqrt(determinant) - np.abs(form_tuple[2])))),
                                   int(np.ceil(np.sqrt(determinant)))):
            if b_prime > np.sqrt(determinant):
                b_prime -= np.abs(form_tuple[2])
            else:
                b_prime += np.abs(form_tuple[2])

        running_form = (form_tuple[2], b_prime, (b_prime ** 2 - determinant) // form_tuple[2])

        # Double check this
        transformation = transformation * Matrix([[0, -1], [1, (form_tuple[1] + running_form[1]) // form_tuple[2]]])

        form_tuple = running_form  # used to be temp form, not running form

        if np.abs(running_form[2]) >= np.abs(running_form[0]):
            break

    # pdb.set_trace()
    return transformation


# Implementation of Art 206
# Vec(Vec(Int, 2), 2) -> Vec(Vec(Int, 2), 2)
def negative_square_determinant_binary_reduction(form):
    determinant = int(form.det())

    if determinant >= 0:
        raise Exception("The determinant of the form", form, "isn't negative.")

    # so we can use Gauss's convention
    determinant *= -1

    if not functools.reduce(lambda x, y: x and y,
                            list(map(lambda x: x % 2 == 0, sp.factorint(determinant).values())),
                            True):
        raise Exception("The negated determinant of the form", form, "is not a perfect square.")

    a = form[0, 0]
    b = form[0, 1]
    c = form[1, 1]

    h = int(np.sqrt(determinant))

    # Recall that Python ranges are inclusive on the left, exclusive on the right
    if a in range(0, 2 * h) and b == h and c == 0:
        raise Exception("Something's wrong. This method shouldn't have been called, right?")

    common_factor = np.gcd(h - b, a)
    beta = (h - b) // common_factor
    delta = a // common_factor

    test, gamma, alpha = xgcd(beta, delta)
    gamma *= -1

    transformation = Matrix([[alpha, beta], [gamma, delta]])

    form_prime = transformation.transpose() * form * transformation

    if form_prime[0] in range(0, 2 * h):
        return transformation

    A = form_prime[0] % (2 * h)

    k = (A - form_prime[0]) // (2 * h)

    return Matrix([[alpha + beta * k, beta], [gamma + delta * k, delta]])


# Implementation of Art 215
# Vec(Vec(Int, 2), 2) -> Vec(Vec(Int, 2), 2)
def zero_determinant_binary_reduction(form):
    if form.det() != 0:
        raise Exception("The determinant of the form", form, "isn't zero.")

    a = form[0, 0]
    b = form[0, 1]
    c = form[1, 1]

    if a == 0:
        raise Exception("Something's wrong. This method shouldn't have been called, right?")

    m = np.gcd(a, c)

    # Why math and not numpy? Because numpy yields a stupid error. It's easier just to use math in this case.
    g = int(math.sqrt(a // m))
    h = int(math.sqrt(c // m))

    if g * h != b // m:
        h *= -1

    _, goth_g, goth_h = xgcd(g, h)

    return Matrix([[h, h + goth_g], [-g, -g + goth_h]])


# Vec(Vec(Int, 2), 2) -> Vec(Vec(Int, 2), 2)
def binary_reduction(form):
    determinant = form.det()

    if determinant > 0:
        # print("POSITIVE DETERMINANT")
        return positive_determinant_binary_reduction(form)
    if determinant < 0:
        if functools.reduce(lambda x, y: x and y,
                            list(map(lambda x: x % 2 == 0, sp.factorint(-determinant).values())),
                            True):
            # print("NEGATIVE QUADRATIC DETERMINANT")
            return negative_square_determinant_binary_reduction(form)
        # print("NEGATIVE NONQUADRATIC DETERMINANT")
        return negative_nonsquare_determinant_binary_reduction(form)
    # print("ZERO DETERMINANT")
    return zero_determinant_binary_reduction(form)


#####################
# Ternary reduction #
#####################

# Vec(Vec(Int, 3), 3) -> Vec(Vec(Int, 3), 3)
def type_two(matrix_q):
    adj_q = -1 * matrix_q.adjugate()
    # first get yz-part of Q
    zy_adj_q = adj_q[1:, 1:]
    # then treat as zy-form
    z_sq_coeff = zy_adj_q[1, 1]
    zy_adj_q[1, 1] = zy_adj_q[0, 0]
    zy_adj_q[0, 0] = z_sq_coeff
    # get 2x2 binary reduction of zy-part
    t = binary_reduction(zy_adj_q)
    # modify into the T that applies to Q
    t[0, 1] = -t[0, 1]
    t[1, 0] = -t[1, 0]
    # t *= -1
    t = t.col_insert(0, sp.Matrix([0, 0]))
    t = t.row_insert(0, sp.Matrix([[1, 0, 0]]))
    return t


# Vec(Vec(Int, 3), 3) -> Vec(Vec(Int, 3), 3)
def type_one(matrix_q):
    xy_q = matrix_q[0:2, 0:2]
    t = binary_reduction(xy_q)
    t = t.col_insert(2, sp.Matrix([0, 0]))
    t = t.row_insert(2, sp.Matrix([[0, 0, 1]]))
    return t


# Vec(Vec(Int, 3), 3) -> Vec(Vec(Int, 3), 3)
def type_iii(q):
    if -1 * q.adjugate()[2, 2] == 0 and q[0, 0] == 0:
        abgcd, cf, cf1 = xgcd(q[1, 1], q[0, 2])
        factor = np.floor(0.5 + (q[1, 2] / abgcd))
        beta = -1 * factor * cf1
        gamma1 = -1 * factor * cf
        gamma = -1 * np.floor(0.5 + ((q[2, 2] + 2 * gamma1 * q[1, 2] + q[1, 1] * gamma1 ** 2) / (2 * q[0, 2])))
    else:
        beta = np.floor(0.5 + (-q[0, 1] / q[0, 0]))
        qad = -1 * q.adjugate()
        gamma1 = np.floor(0.5 + (qad[1, 2] / qad[2, 2]))
        gamma = np.floor(0.5 + ((qad[0, 2] + qad[2, 2] * beta * gamma1 - qad[1, 2] * beta) / qad[2, 2]))
    a = Matrix(3, 3, lambda i, j: 1 if i == j else 0)
    a[0, 1] = beta
    a[0, 2] = gamma
    a[1, 2] = gamma1
    return a


# Vec(Vec(Int, 3), 3) -> Vec(Vec(Int, 3), 3)
def oscillate(matrix_q):
    adj_q = -1 * matrix_q.adjugate()
    aD = (-1 * adj_q.adjugate())[0, 0]
    t = Matrix.eye(3)
    # check for end of descent process
    while not (abs(matrix_q[0, 0]) <= sp.sqrt((4 / 3) * abs(adj_q[2, 2])) and
               abs(adj_q[2, 2]) <= sp.sqrt((4 / 3) * abs(matrix_q[0, 0] * aD))):
        temp = None
        if abs(matrix_q[0, 0]) > sp.sqrt((4 / 3) * abs(adj_q[2, 2])):
            temp = type_one(matrix_q)
        # even if neither condition holds, only does one type per loop
        else:
            temp = type_two(matrix_q)
        if t is None:
            t = temp
        else:
            t = t * temp
        # update matrices before checking conditions
        matrix_q = temp.T * matrix_q * temp
        adj_q = -1 * matrix_q.adjugate()
        aD = -1 * adj_q.adjugate()[0, 0]
    return t * type_iii(matrix_q)


# Art 277
# Vec(Vec(Int, 3), 3) -> Vec(Vec(Int, 3), 3)
def get_final_transformation(form):
    gauss_form = matrix_to_gauss(form)

    rotator = Matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    if gauss_form == (0, 1, 0, 0, -1, 0):
        intermediate = Matrix([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, -1]])
    elif gauss_form == (0, 1, 1, 0, 1, 0):
        intermediate = Matrix([[0, 0, 1],
                               [0, 1, -1],
                               [1, 1, 0]])
    elif gauss_form == (0, 1, 1, 0, -1, 0):
        intermediate = Matrix([[0, 0, 1],
                               [0, 1, -1],
                               [-1, 1, 0]])
    elif gauss_form == (0, 1, -1, 0, 1, 0):
        intermediate = Matrix([[0, 0, 1],
                               [0, 1, 1],
                               [1, -1, -1]])
    elif gauss_form == (0, 1, -1, 0, -1, 0):
        intermediate = Matrix([[0, 0, 1],
                               [0, 1, 1],
                               [-1, -1, -1]])
    elif gauss_form == (1, 1, -1, 0, 0, 0):
        intermediate = Matrix([[1, 0, -1],
                               [1, 1, -1],
                               [0, -1, 1]])
    elif gauss_form == (-1, 1, 1, 0, 0, 0):
        intermediate = Matrix([[1, 0, -1],
                               [1, 1, -1],
                               [0, -1, 1]]) * rotator
    elif gauss_form == (1, -1, 1, 0, 0, 0):
        intermediate = Matrix([[1, 0, -1],
                               [1, 1, -1],
                               [0, -1, 1]]) * rotator * rotator
    else:
        intermediate = Matrix.eye(3)

    return (Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 1]]) * intermediate).inv()


###########
# g stuff #
###########

def get_S_prime(form):
    xform = oscillate(form)
    return xform * get_final_transformation(xform.T * form * xform)


################
# Do the thing #
################

# Int -> Int -> Int -> List(Vec(Int, 3))
def get_rational_point(a, b, c, strict_squarefree=False):
    if 0 in [a, b, c]:
        raise Exception("Parameters must be nonzero")

    reduced_parameters = reduce(a, b, c)
    if reduced_parameters != (a, b, c):
        a, b, c = reduced_parameters
        print("Provided parameters are not coprime. Attempting to use", reduced_parameters, "instead.")

    if not strict_squarefree:
        reduced_parameters = remove_squares(a, b, c)
        if reduced_parameters != (a, b, c):
            a, b, c = reduced_parameters
            print("Reduced parameters aren't square-free. Attempting to use", reduced_parameters, "instead.")

    check_reduced_params(a, b, c, strict_squarefree)

    goth_a, goth_b, goth_c = calculate_root(b, c, a), calculate_root(a, c, b), calculate_root(a, b, c)
    A, B, C = find_ABC(a, b, c, goth_a, goth_b, goth_c)

    g_transformation = get_g_transformation(A * a, B * b, C * c)
    g = g_transformation.T * gauss_to_matrix(a, b, c, 0, 0, 0) * g_transformation
    (m, m1, m2, n, n1, n2) = matrix_to_gauss(g)

    d = -a * b * c  # determinant in Gauss's convention

    S = copy.deepcopy(g_transformation)  # A copy isn't necessary, doing it for clarity/debugging convenience
    S[0, 0] *= d
    S[1, 0] *= d
    S[2, 0] *= d

    (M, M1, M2, N, N1, N2) = (m * d, m1 // d, m2 // d, n // d, n1, n2)

    S_prime = get_S_prime(gauss_to_matrix(M, M1, M2, N, N1, N2))

    final = S * S_prime

    return [reduce(x, y, z) for (x, y, z) in (list(final.col(1)), list(final.col(2)))]


# Int -> Int -> Int -> Vec(Vec(Int, 3), 1)
def get_rational_point_on_conic(a, b, c):
    form_rational_point = get_rational_point(4 * a ** 2,
                                             4 * a * c - b ** 2,
                                             -4 * a,
                                             strict_squarefree=True)[0]

    if form_rational_point[0][2] != 0:
        x_prime, y, z = form_rational_point[0]
        x = x_prime - (b / (2 * a)) * y
    elif form_rational_point[1][2] != 0:
        x_prime, y, z = form_rational_point[1]
        x = x_prime - (b / (2 * a)) * y
    else:
        raise Exception("Gauss' method returns z = 0.")

    return x / z, y / z


####################
# Convenient tools #
####################

# Int -> Int -> Int -> Int -> Int -> Int -> Vec(Vec(Int, 3), 3)
def gauss_to_matrix(a, b, c, d, e, f):
    return (Matrix([[a, f, e],
                    [f, b, d],
                    [e, d, c]]))


# Vec(Vec(Int, 3), 3) -> (Int, Int, Int, Int, Int, Int)
def matrix_to_gauss(form):
    if form.T != form:
        raise Exception("The provided form isn't symmetric.")

    return form[0, 0], form[1, 1], form[2, 2], form[1, 2], form[0, 2], form[0, 1]


#################
# Testing stuff #
#################

# Int -> Int -> Int -> Bool
def check_reduced_params_bool(a, b, c):
    if len(set([x < 0 for x in [a, b, c]])) < 2:
        return False
    if not are_pairwise_coprime([a, b, c]):
        return False
    if not functools.reduce(lambda x, y: x and y, list(map(is_squarefree, [a, b, c]))):
        return False
    if not is_quadratic_residue(-b * c, a):
        return False
    if not is_quadratic_residue(-a * c, b):
        return False
    if not is_quadratic_residue(-a * b, c):
        return False
    return True
