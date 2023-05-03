import jax
import jax.numpy as jnp

def rotmat_to_qtn(R):
    tr = jnp.trace(R)
    def normal(R, tr):
        s = jnp.sqrt(tr+1.)*2
        w = 0.25*s
        x = (R[2,1] - R[1,2])/s
        y = (R[0,2] - R[2,0])/s
        z = (R[1,0] - R[0,1])/s
        return jnp.array([x, y, z, w])
    def case1(R, tr):
        s= jnp.sqrt(1. + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s; 
        z = (R[0,2] + R[2,0]) / s; 
        return jnp.array([x, y, z, w])
    def case2(R, tr):
        s= jnp.sqrt(1. + R[1,1] - R[0,0] - R[2,2]) * 2 
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
        return jnp.array([x, y, z, w])
    def case3(R, tr):
        s = jnp.sqrt(1. + R[2,2] - R[1,1] - R[0,0]) * 2
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
        return jnp.array([x, y, z, w])
    is_normal = tr > 0.
    is_case1 = (R[0,0]>R[1,1]) & (R[0,0] > R[2,2])
    is_case2 = R[1,1] > R[2,2]
    is_case3 = ~is_normal & ~is_case1 & ~is_case2
    switch = is_case1 + is_case2 * 2 + is_case3 * 3
    return jax.lax.switch(switch, (normal, case1, case2, case3), R, tr)

rotmat_to_qtn_batch = jax.jit(jax.vmap(rotmat_to_qtn))
rotmat_to_qtn = jax.jit(rotmat_to_qtn)
