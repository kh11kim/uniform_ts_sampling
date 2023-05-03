import numpy as np

def primes(nb_primes):
    if nb_primes > 1000:
        nb_primes = 1000
    
    p = np.zeros(nb_primes)
    len_p = 0
    n = 2
    while len_p < nb_primes:
        for i in p[:len_p]:
            if n % i == 0:
                break
            
        else:
            p[len_p] = n
            len_p += 1
        n += 1
    result_as_list = [prime for prime in p[:len_p]]
    return result_as_list


