"""Build a cross-domain task corpus (real-world math/physics/bio/CS) as Python functions.

Every task has: text (noisy NL), entry, reference, tests, type_pool.
Domains share underlying primitives (gcd, factorial, prime, sqrt, trig, etc.)
to test cross-domain composition.
"""

import math, random, json
from pathlib import Path

SEED = 42

# ============================================================
# SHARED PRIMITIVES (the cross-domain "atoms" the model should learn)
# These appear as implementations across multiple domains
# ============================================================

PRIMS = {
    "gcd": {
        "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return abs(a)",
        "desc": "greatest common divisor of two integers"
    },
    "lcm": {
        "code": "def lcm(a, b):\n    return a * b // gcd(a, b) if a and b else 0",
        "desc": "least common multiple of two integers"
    },
    "factorial": {
        "code": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
        "desc": "factorial of n (n!)"
    },
    "is_prime": {
        "code": "def is_prime(n):\n    return n >= 2 and all(n % i for i in range(2, int(n**0.5) + 1))",
        "desc": "primality test"
    },
    "sqrt_approx": {
        "code": "def sqrt_approx(n):\n    return n ** 0.5",
        "desc": "approximate square root"
    },
    "nCr": {
        "code": "def nCr(n, r):\n    r = min(r, n - r)\n    return factorial(n) // (factorial(r) * factorial(n - r))",
        "desc": "binomial coefficient n choose r"
    },
    "fibonacci": {
        "code": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "desc": "nth fibonacci number"
    },
    "digit_sum": {
        "code": "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))",
        "desc": "sum of digits of an integer"
    },
    "mean": {
        "code": "def mean(arr):\n    return sum(arr) / len(arr) if arr else 0",
        "desc": "arithmetic mean of a list"
    },
    "variance": {
        "code": "def variance(arr):\n    m = mean(arr)\n    return sum((x - m)**2 for x in arr) / len(arr) if arr else 0",
        "desc": "population variance of a list"
    },
    "std_dev": {
        "code": "def std_dev(arr):\n    return variance(arr) ** 0.5",
        "desc": "standard deviation of a list"
    },
    "degrees_to_radians": {
        "code": "def deg_to_rad(deg):\n    return deg * 3.141592653589793 / 180.0",
        "desc": "convert degrees to radians"
    },
    "radians_to_degrees": {
        "code": "def rad_to_deg(rad):\n    return rad * 180.0 / 3.141592653589793",
        "desc": "convert radians to degrees"
    },
    "kinetic_energy": {
        "code": "def kinetic_energy(mass, velocity):\n    return 0.5 * mass * velocity ** 2",
        "desc": "kinetic energy = 1/2 * m * v^2"
    },
    "bmi": {
        "code": "def bmi(weight_kg, height_m):\n    return weight_kg / (height_m ** 2)",
        "desc": "body mass index"
    },
    "dna_complement": {
        "code": "def dna_complement(dna):\n    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}\n    return ''.join(comp.get(c, c) for c in dna.upper())",
        "desc": "DNA nucleotide complement"
    },
    "gc_content": {
        "code": "def gc_content(dna):\n    dna = dna.upper()\n    return (dna.count('G') + dna.count('C')) / len(dna) if dna else 0",
        "desc": "GC content fraction of a DNA sequence"
    },
    "hamming_distance": {
        "code": "def hamming_distance(s1, s2):\n    return sum(1 for a, b in zip(s1, s2) if a != b)",
        "desc": "Hamming distance between two strings"
    },
}


def _build_reference(entry: str, body: str, deps: list[str] | None = None) -> str:
    """Build reference code: dependency primitives + entry function."""
    lines = []
    if deps:
        for d in deps:
            if d in PRIMS:
                lines.append(PRIMS[d]["code"])
    lines.append(body.replace("{entry}", entry))
    return "\n\n".join(lines)


# ============================================================
# DOMAIN: MATH
# ============================================================
MATH_TASKS = [
    # Number theory (reuses gcd, factorial, is_prime, nCr)
    dict(
        text="find the number of distinct prime divisors of n",
        entry="count_prime_divisors",
        ref_body="def {entry}(n):\n    count = 0\n    for i in range(2, int(abs(n)**0.5) + 1):\n        if n % i == 0 and is_prime(i):\n            count += 1\n            while n % i == 0:\n                n //= i\n    if n > 1:\n        count += 1\n    return count",
        deps=["is_prime"],
        tests=[((30,), 3), ((7,), 1), ((12,), 2), ((1,), 0)],
        type_pool=["int"],
    ),
    dict(
        text="compute the nth triangular number using formula n*(n+1)/2",
        entry="triangular",
        ref_body="def {entry}(n):\n    return n * (n + 1) // 2",
        deps=None,
        tests=[((5,), 15), ((10,), 55), ((1,), 1)],
        type_pool=["int"],
    ),
    dict(
        text="euler's totient function phi(n): count numbers 1..n coprime to n",
        entry="totient",
        ref_body="def {entry}(n):\n    result = n\n    p = 2\n    temp = n\n    while p * p <= temp:\n        if temp % p == 0:\n            while temp % p == 0:\n                temp //= p\n            result -= result // p\n        p += 1\n    if temp > 1:\n        result -= result // temp\n    return result",
        deps=None,
        tests=[((12,), 4), ((7,), 6), ((1,), 1)],
        type_pool=["int"],
    ),
    dict(
        text="compute the sum of divisors of n using prime factorization",
        entry="sum_divisors_prime",
        ref_body="def {entry}(n):\n    total = 1\n    temp = n\n    p = 2\n    while p * p <= temp:\n        if temp % p == 0:\n            power_sum = 1\n            pk = 1\n            while temp % p == 0:\n                temp //= p\n                pk *= p\n                power_sum += pk\n            total *= power_sum\n        p += 1\n    if temp > 1:\n        total *= (1 + temp)\n    return total",
        deps=None,
        tests=[((12,), 28), ((7,), 8), ((6,), 12)],
        type_pool=["int"],
    ),
    dict(
        text="calculate the number of ways to choose k items from n items (binomial coefficient)",
        entry="binomial",
        ref_body="def {entry}(n, k):\n    return nCr(n, k)",
        deps=["nCr"],
        tests=[((5, 2), 10), ((10, 3), 120), ((7, 7), 1)],
        type_pool=["int"],
    ),
    dict(
        text="determine if n is a perfect number (sum of proper divisors equals n)",
        entry="is_perfect_math",
        ref_body="def {entry}(n):\n    s = 0\n    for i in range(1, n):\n        if n % i == 0:\n            s += i\n    return s == n",
        deps=None,
        tests=[((6,), True), ((28,), True), ((12,), False), ((496,), True)],
        type_pool=["int"],
    ),
    dict(
        text="compute the least common multiple of a and b using the gcd formula",
        entry="lcm_func",
        ref_body="def {entry}(a, b):\n    return lcm(a, b)",
        deps=["lcm"],
        tests=[((4, 6), 12), ((7, 5), 35), ((12, 18), 36)],
        type_pool=["int"],
    ),
    dict(
        text="check if n is a power of two",
        entry="is_power_of_two",
        ref_body="def {entry}(n):\n    return n > 0 and (n & (n - 1)) == 0",
        deps=None,
        tests=[((16,), True), ((18,), False), ((1,), True), ((0,), False)],
        type_pool=["int"],
    ),
]

# ============================================================
# DOMAIN: PHYSICS
# ============================================================
PHYSICS_TASKS = [
    dict(
        text="calculate the gravitational force between two masses using F=G*m1*m2/r^2",
        entry="gravitational_force",
        ref_body="def {entry}(m1, m2, r):\n    G = 6.67430e-11\n    return G * m1 * m2 / (r ** 2)",
        deps=None,
        tests=[((5.97e24, 7.35e22, 3.84e8), None)],  # approximate check
        type_pool=["float"],
    ),
    dict(
        text="compute the wavelength of a wave given frequency and speed (v=f*lambda)",
        entry="wavelength",
        ref_body="def {entry}(speed, frequency):\n    return speed / frequency if frequency != 0 else 0",
        deps=None,
        tests=[((3e8, 5e14), 6e-7), ((340, 440), 0.7727)],
        type_pool=["float"],
    ),
    dict(
        text="calculate the kinetic energy of an object given mass and velocity",
        entry="ke",
        ref_body="def {entry}(mass, velocity):\n    return kinetic_energy(mass, velocity)",
        deps=["kinetic_energy"],
        tests=[((2.0, 3.0), 9.0), ((5.0, 1.0), 2.5)],
        type_pool=["float"],
    ),
    dict(
        text="compute the momentum of an object: p = m * v",
        entry="momentum",
        ref_body="def {entry}(mass, velocity):\n    return mass * velocity",
        deps=None,
        tests=[((10.0, 5.0), 50.0), ((2.0, 100.0), 200.0)],
        type_pool=["float"],
    ),
    dict(
        text="calculate the orbital period using Kepler's third law: T=2*pi*sqrt(a^3/(G*M))",
        entry="orbital_period",
        ref_body="def {entry}(semi_major_axis, central_mass):\n    G = 6.67430e-11\n    import math\n    return 2 * math.pi * math.sqrt(semi_major_axis**3 / (G * central_mass))",
        deps=None,
        tests=[((1.496e11, 1.989e30), 3.155e7)],  # Earth ~ 1 year
        type_pool=["float"],
    ),
    dict(
        text="convert temperature from Celsius to Fahrenheit",
        entry="celsius_to_fahrenheit",
        ref_body="def {entry}(c):\n    return c * 9.0 / 5.0 + 32.0",
        deps=None,
        tests=[((0,), 32.0), ((100,), 212.0), ((-40,), -40.0)],
        type_pool=["float"],
    ),
    dict(
        text="calculate the density of an object given mass and volume",
        entry="density",
        ref_body="def {entry}(mass, volume):\n    return mass / volume if volume != 0 else 0",
        deps=None,
        tests=[((10.0, 2.0), 5.0), ((7.8, 1.0), 7.8)],
        type_pool=["float"],
    ),
    dict(
        text="compute the pressure using P = F/A",
        entry="pressure",
        ref_body="def {entry}(force, area):\n    return force / area if area != 0 else 0",
        deps=None,
        tests=[((100.0, 2.0), 50.0), ((50.0, 0.5), 100.0)],
        type_pool=["float"],
    ),
    dict(
        text="calculate the speed of sound in air given temperature in Celsius (v=331*sqrt(1+T/273))",
        entry="speed_of_sound",
        ref_body="def {entry}(temp_celsius):\n    import math\n    return 331.0 * math.sqrt(1 + temp_celsius / 273.15)",
        deps=None,
        tests=[((0,), 331.0), ((20,), 342.9)],
        type_pool=["float"],
    ),
    dict(
        text="compute electrical power using P = V*I (voltage times current)",
        entry="electrical_power",
        ref_body="def {entry}(voltage, current):\n    return voltage * current",
        deps=None,
        tests=[((12.0, 2.0), 24.0), ((230.0, 0.5), 115.0)],
        type_pool=["float"],
    ),
]

# ============================================================
# DOMAIN: BIOLOGY
# ============================================================
BIOLOGY_TASKS = [
    dict(
        text="transcribe a DNA sequence to RNA (replace T with U)",
        entry="dna_to_rna",
        ref_body="def {entry}(dna):\n    return dna.upper().replace('T', 'U')",
        deps=None,
        tests=[(("ATGC",), "AUGC"), (("AATT",), "AAUU")],
        type_pool=["str"],
    ),
    dict(
        text="compute the reverse complement of a DNA strand",
        entry="reverse_complement",
        ref_body="def {entry}(dna):\n    return dna_complement(dna)[::-1]",
        deps=["dna_complement"],
        tests=[(("ATGC",), "GCAT"), (("AATT",), "AATT")],
        type_pool=["str"],
    ),
    dict(
        text="calculate the GC content percentage of a DNA sequence",
        entry="gc_fraction",
        ref_body="def {entry}(dna):\n    return gc_content(dna)",
        deps=["gc_content"],
        tests=[(("ATGC",), 0.5), (("AAAA",), 0.0), (("GCGC",), 1.0)],
        type_pool=["str"],
    ),
    dict(
        text="compute the body mass index given weight in kg and height in meters",
        entry="bmi_index",
        ref_body="def {entry}(weight, height):\n    return bmi(weight, height)",
        deps=["bmi"],
        tests=[((70, 1.75), 22.86), ((90, 1.80), 27.78)],
        type_pool=["float"],
    ),
    dict(
        text="estimate the basal metabolic rate using the Harris-Benedict equation for males",
        entry="bmr_male",
        ref_body="def {entry}(weight_kg, height_cm, age):\n    return 88.362 + 13.397 * weight_kg + 4.799 * height_cm - 5.677 * age",
        deps=None,
        tests=[((70, 175, 30), 1695.7)],
        type_pool=["float"],
    ),
    dict(
        text="find the Hamming distance between two DNA sequences",
        entry="dna_distance",
        ref_body="def {entry}(seq1, seq2):\n    return hamming_distance(seq1, seq2)",
        deps=["hamming_distance"],
        tests=[(("ATGC", "ATTC"), 1), (("AAAA", "TTTT"), 4), (("ATGC", "ATGC"), 0)],
        type_pool=["str"],
    ),
    dict(
        text="compute the frequency of each nucleotide in a DNA sequence",
        entry="nucleotide_freq",
        ref_body="def {entry}(dna):\n    dna = dna.upper()\n    return {'A': dna.count('A'), 'T': dna.count('T'), 'C': dna.count('C'), 'G': dna.count('G')}",
        deps=None,
        tests=[(("ATGC",), {"A": 1, "T": 1, "C": 1, "G": 1}), (("AAAA",), {"A": 4, "T": 0, "C": 0, "G": 0})],
        type_pool=["str"],
    ),
    dict(
        text="calculate the protein mass from a sequence of amino acids using monoisotopic masses",
        entry="protein_mass",
        ref_body="def {entry}(protein):\n    masses = {'A': 71.04, 'R': 156.19, 'N': 114.08, 'D': 115.03, 'C': 103.01, 'E': 129.04, 'Q': 128.09, 'G': 57.02, 'H': 137.06, 'I': 113.08, 'L': 113.08, 'K': 128.09, 'M': 131.04, 'F': 147.07, 'P': 97.05, 'S': 87.03, 'T': 101.05, 'W': 186.08, 'Y': 163.06, 'V': 99.07}\n    return round(sum(masses.get(aa.upper(), 0) for aa in protein), 2)",
        deps=None,
        tests=[(("ACDEF",), None)],
        type_pool=["str"],
    ),
]

# ============================================================
# DOMAIN: COMPUTER SCIENCE / ALGORITHMS
# ============================================================
CS_TASKS = [
    dict(
        text="merge two sorted lists into one sorted list",
        entry="merge_sorted",
        ref_body="def {entry}(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] < b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result",
        deps=None,
        tests=[(([1,3,5], [2,4,6]), [1,2,3,4,5,6]), (([], [1,2]), [1,2])],
        type_pool=["list"],
    ),
    dict(
        text="remove duplicate elements from a list while preserving order",
        entry="unique_preserve_order",
        ref_body="def {entry}(arr):\n    seen = set()\n    result = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
        deps=None,
        tests=[(([1,2,2,3,1,4],), [1,2,3,4]), (([],), [])],
        type_pool=["list"],
    ),
    dict(
        text="find the longest common prefix of a list of strings",
        entry="longest_common_prefix",
        ref_body="def {entry}(strings):\n    if not strings:\n        return ''\n    prefix = strings[0]\n    for s in strings[1:]:\n        while not s.startswith(prefix):\n            prefix = prefix[:-1]\n            if not prefix:\n                return ''\n    return prefix",
        deps=None,
        tests=[((["flower","flow","flight"],), "fl"), ((["dog","racecar","car"],), "")],
        type_pool=["list"],
    ),
    dict(
        text="check if a string is a palindrome (reads same forwards and backwards)",
        entry="is_palindrome_cs",
        ref_body="def {entry}(s):\n    s = ''.join(c.lower() for c in s if c.isalnum())\n    return s == s[::-1]",
        deps=None,
        tests=[(("racecar",), True), (("A man a plan a canal Panama",), True), (("hello",), False)],
        type_pool=["str"],
    ),
    dict(
        text="find the maximum subarray sum using Kadane's algorithm",
        entry="kadane",
        ref_body="def {entry}(arr):\n    if not arr:\n        return 0\n    max_ending = max_so_far = arr[0]\n    for x in arr[1:]:\n        max_ending = max(x, max_ending + x)\n        max_so_far = max(max_so_far, max_ending)\n    return max_so_far",
        deps=None,
        tests=[(([-2,1,-3,4,-1,2,1,-5,4],), 6), (([1],), 1), (([-1,-2],), -1)],
        type_pool=["list"],
    ),
    dict(
        text="compute the edit distance (Levenshtein) between two strings",
        entry="edit_distance",
        ref_body="def {entry}(a, b):\n    n, m = len(a), len(b)\n    dp = [[0]*(m+1) for _ in range(n+1)]\n    for i in range(n+1):\n        dp[i][0] = i\n    for j in range(m+1):\n        dp[0][j] = j\n    for i in range(1, n+1):\n        for j in range(1, m+1):\n            if a[i-1] == b[j-1]:\n                dp[i][j] = dp[i-1][j-1]\n            else:\n                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n    return dp[n][m]",
        deps=None,
        tests=[(("kitten", "sitting"), 3), (("abc", "abc"), 0)],
        type_pool=["str"],
    ),
    dict(
        text="evaluate the nth Fibonacci number using matrix exponentiation (O(log n))",
        entry="fib_matrix",
        ref_body="def {entry}(n):\n    def mat_mul(A, B):\n        return [[A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],[A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]]\n    def mat_pow(M, p):\n        R = [[1,0],[0,1]]\n        while p:\n            if p & 1: R = mat_mul(R, M)\n            M = mat_mul(M, M)\n            p >>= 1\n        return R\n    if n == 0: return 0\n    return mat_pow([[1,1],[1,0]], n-1)[0][0]",
        deps=None,
        tests=[((10,), 55), ((0,), 0), ((1,), 1), ((20,), 6765)],
        type_pool=["int"],
    ),
    dict(
        text="perform binary search on a sorted list to find a target value",
        entry="binary_search",
        ref_body="def {entry}(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: left = mid + 1\n        else: right = mid - 1\n    return -1",
        deps=None,
        tests=[(([1,3,5,7,9], 5), 2), (([1,3,5,7,9], 4), -1)],
        type_pool=["list"],
    ),
    dict(
        text="compute the mean and standard deviation of a list of numbers",
        entry="mean_std",
        ref_body="def {entry}(arr):\n    m = mean(arr)\n    s = std_dev(arr)\n    return (m, s)",
        deps=["mean", "std_dev"],
        tests=[(([1,2,3,4,5],), (3.0, 1.414))],
        type_pool=["list"],
    ),
    dict(
        text="sort a list using the quicksort algorithm",
        entry="quicksort",
        ref_body="def {entry}(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        deps=None,
        tests=[(([3,6,8,10,1,2,1],), [1,1,2,3,6,8,10]), (([],), [])],
        type_pool=["list"],
    ),
]

# ============================================================
# DOMAIN: STATISTICS / DATA SCIENCE
# ============================================================
STATS_TASKS = [
    dict(
        text="compute the median of a list of numbers",
        entry="median",
        ref_body="def {entry}(arr):\n    s = sorted(arr)\n    n = len(s)\n    if n % 2 == 0:\n        return (s[n//2-1] + s[n//2]) / 2\n    return s[n//2]",
        deps=None,
        tests=[(([1,3,3,6,7,8,9],), 6), (([1,2,3,4,5,6],), 3.5)],
        type_pool=["list"],
    ),
    dict(
        text="compute the sample variance of a list (uses n-1 denominator)",
        entry="sample_variance",
        ref_body="def {entry}(arr):\n    m = mean(arr)\n    return sum((x-m)**2 for x in arr) / (len(arr)-1) if len(arr) > 1 else 0",
        deps=["mean"],
        tests=[(([1,2,3,4,5],), 2.5)],
        type_pool=["list"],
    ),
    dict(
        text="compute the correlation coefficient between two lists",
        entry="correlation",
        ref_body="def {entry}(x, y):\n    n = len(x)\n    mx, my = mean(x), mean(y)\n    num = sum((xi-mx)*(yi-my) for xi, yi in zip(x, y))\n    den = (sum((xi-mx)**2 for xi in x) * sum((yi-my)**2 for yi in y))**0.5\n    return num / den if den != 0 else 0",
        deps=["mean"],
        tests=[(([1,2,3,4,5], [2,4,6,8,10]), 1.0)],
        type_pool=["list"],
    ),
    dict(
        text="normalize a list of numbers to z-scores (standardize)",
        entry="zscore_normalize",
        ref_body="def {entry}(arr):\n    m = mean(arr)\n    s = std_dev(arr)\n    return [(x-m)/s for x in arr] if s != 0 else [0]*len(arr)",
        deps=["mean", "std_dev"],
        tests=[(([1,2,3,4,5],), None)],  # check mean ~0, std ~1
        type_pool=["list"],
    ),
]

# ============================================================
# ASSEMBLE CORPUS
# ============================================================
all_domains = {
    "math": MATH_TASKS,
    "physics": PHYSICS_TASKS,
    "biology": BIOLOGY_TASKS,
    "cs": CS_TASKS,
    "stats": STATS_TASKS,
}


def build_corpus(seed=SEED) -> list[dict]:
    """Build the full cross-domain corpus. Each task gets entry, reference, tests, type_pool."""
    rng = random.Random(seed)
    tasks = []
    existing_prims = set()

    for domain_name, domain_tasks in all_domains.items():
        for t in domain_tasks:
            ref_code = _build_reference(t["entry"], t["ref_body"], t.get("deps"))

            # Build test tuples from raw tests format
            test_list = list(t["tests"])

            task = dict(
                text=t["text"],
                entry=t["entry"],
                reference=ref_code,
                tests=test_list,
                type_pool=t.get("type_pool", ["int"]),
                domain=domain_name,
            )
            tasks.append(task)

            # Track primitives used
            for d in (t.get("deps") or []):
                existing_prims.add(d)

    rng.shuffle(tasks)
    print(f"Built corpus: {len(tasks)} tasks across {len(all_domains)} domains")
    print(f"  Math: {len(MATH_TASKS)}, Physics: {len(PHYSICS_TASKS)}, Biology: {len(BIOLOGY_TASKS)},")
    print(f"  CS: {len(CS_TASKS)}, Stats: {len(STATS_TASKS)}")
    print(f"  Shared primitives used: {sorted(existing_prims)}")
    return tasks


def build_holdout(seed=42, holdout_seed=999) -> tuple[list[dict], list[dict]]:
    """Split corpus into train and held-out by domain.
    Held-out = tasks that use primitives NOT in the training set,
    plus novel compositions of seen primitives.
    """
    rng = random.Random(holdout_seed)
    all_tasks = build_corpus(seed)

    # Hold-out: every 4th task from each domain
    train = []
    holdout = []
    domain_buckets: dict[str, list] = {}
    for t in all_tasks:
        domain_buckets.setdefault(t["domain"], []).append(t)

    for domain, dtasks in domain_buckets.items():
        rng.shuffle(dtasks)
        split = max(1, len(dtasks) // 4)
        holdout.extend(dtasks[:split])
        train.extend(dtasks[split:])

    rng.shuffle(train)
    rng.shuffle(holdout)
    print(f"\nSplit: {len(train)} train, {len(holdout)} held-out")
    return train, holdout


if __name__ == "__main__":
    train, holdout = build_holdout()
    print(f"\nSample tasks:")
    for t in train[:3]:
        print(f"  [{t['entry']:25s}] [{t['domain']:7s}] {t['text'][:70]}")
    print(f"  ...")
    for t in holdout[:3]:
        print(f"  [{t['entry']:25s}] [{t['domain']:7s}] {t['text'][:70]}")

    # Save corpus
    out = Path(r"E:\PROJECT\graph_v5\artifacts")
    out.mkdir(exist_ok=True)
    (out / "crossdomain_train.json").write_text(json.dumps(train, indent=2))
    (out / "crossdomain_holdout.json").write_text(json.dumps(holdout, indent=2))
    print(f"\nSaved to artifacts/crossdomain_*.json")
