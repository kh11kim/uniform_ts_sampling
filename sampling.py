import numpy as np

def generate_uniform_points_on_sphere(num_points):
    """spherical coordinate"""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    points = np.zeros((num_points, 3))
    points[:, 0] = np.cos(theta) * np.sin(phi)
    points[:, 1] = np.sin(theta) * np.sin(phi)
    points[:, 2] = np.cos(phi)
    return points

def generate_qtn_grid(n):
    """This code utilizes fibonacci spiral """
    phi = np.sqrt(2)
    psi = 1.533751168755204288118041 
    qtns = []
    for i in range(n):
        s = i + 1/2
        t = s/n
        d = np.pi * 2 * s
        r = np.sqrt(t)
        R = np.sqrt(1-t)
        alpha, beta = d/phi, d/psi
        qtn = np.array([r*np.sin(alpha), r*np.cos(alpha), R*np.sin(beta), R*np.cos(beta)])
        qtns.append(qtn)
    return qtns