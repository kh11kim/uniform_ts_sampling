import jax.numpy as jnp
from jax import lax, jit

def vee(so3mat):
    return jnp.array([so3mat[2,1], so3mat[0,2], so3mat[1,0]])
    
def ec(R):
    def zero_angle(acosinput, R):
        return jnp.zeros(3)
    def pi_angle(acosinput, R):
        def get_omg_case1(R):
            return (1.0 / jnp.sqrt(2 * (1 + R[2,2]))) \
                  * jnp.array([R[0,2], R[1,2], 1 + R[2,2]])
        def get_omg_case2(R):
            return (1.0 / jnp.sqrt(2 * (1 + R[1,1]))) \
                  * jnp.array([R[0,1], 1 + R[1,1], R[2,1]])
        def get_omg_case0(R):
            return (1.0 / jnp.sqrt(2 * (1 + R[0,0]))) \
                  * jnp.array([1 + R[0,0], R[1,0], R[2,0]])
        case1 = jnp.abs(1 + R[2,2]) >= 1e-10
        case2 = jnp.abs(1 + R[1,1]) >= 1e-10
        case = case1 + case2*2
        return jnp.pi * lax.switch(case, (get_omg_case0, get_omg_case1, get_omg_case2), R)
    def normal_case(acosinput, R):
        angle = jnp.arccos(acosinput)
        return angle / 2. / jnp.sin(angle) * vee(R - jnp.array(R).T)
    acosinput = (jnp.trace(R) - 1.) / 2.0
    is_zero_angle = acosinput >= 1.
    is_pi_angle = acosinput <= -1.
    cond = (is_zero_angle + is_pi_angle*2).astype(int)
    return lax.switch(cond, (normal_case, zero_angle, pi_angle), acosinput, R)

ec = jit(ec)