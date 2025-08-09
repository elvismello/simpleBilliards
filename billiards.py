
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib

plt.style.use('dark_background')

class simple_particle:

    def __init__ (self, mass, pos, vel, color='#1f77b4'):
        self.mass = mass
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.acel = np.array([0.0, 0.0, 0.0])
        self.color = color



class non_rotating_pool_ball(simple_particle):

    def __init__ (self, radius, mass, pos, vel, color='#1f77b4'):
        self.mass = mass
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.acel = np.array([0.0, 0.0, 0.0])
        self.color = color
        self.radius = radius
        self.is_colliding = False
        self.last_collision = 0



def detect_collision(ball_1, ball_2):
    return np.linalg.norm(ball_1.pos - ball_2.pos) <= (ball_1.radius + ball_2.radius)
    


def detect_collision_wall(ball, bounds):
    return np.any(ball.pos - ball.radius < 0) or np.any(ball.pos + ball.radius > bounds)



def get_collision_normal(ball_1, ball_2):

    vector = ball_1.pos - ball_2.pos
    unit_vector = vector / np.linalg.norm(vector)
    
    return unit_vector



def get_collision_normal_wall(ball, bounds):

    # is it in X
    if (ball.pos[0] - ball.radius <= 0) or (ball.pos[0] + ball.radius >= bounds):
        if (ball.pos[0] - ball.radius <= 0):
            collision_normal = np.array([1.0, 0.0, 0.0])
        else:
            collision_normal = np.array([1.0, 0.0, 0.0])
    # is it in Y
    elif (ball.pos[1] - ball.radius <= 0) or (ball.pos[1] + ball.radius >= bounds):
        if (ball.pos[1] - ball.radius <= 0):
            collision_normal = np.array([0.0, 1.0, 0.0])
        else:
            collision_normal = np.array([0.0, 1.0, 0.0])
        
    # is it in Z
    else:
        if (ball.pos[2] - ball.radius <= 0):
            collision_normal = np.array([0.0, 0.0, 1.0])
        else:
            collision_normal = np.array([0.0, 0.0, 1.0])
    
    return collision_normal



def project_vectors(x, y):
    """ projects x onto y """
    return y * np.dot(x, y) / np.dot(y, y)



def calculate_collision(ball_1, ball_2, frame, dt):

    if True:    
        collision_normal = get_collision_normal(ball_1, ball_2)
    
        ball_1_parallel = project_vectors(ball_1.vel, collision_normal)
        ball_1_perpendicular = ball_1.vel - ball_1_parallel
    
        ball_2_parallel = project_vectors(ball_2.vel, collision_normal)
        ball_2_perpendicular = ball_2.vel - ball_2_parallel
    
        mass_sum = ball_1.mass + ball_2.mass
    
        ball_1_parallel_final = (ball_1_parallel * (ball_1.mass - ball_2.mass) + 2 * ball_2.mass * ball_2_parallel) / mass_sum
        ball_2_parallel_final = (ball_2_parallel * (ball_2.mass - ball_1.mass) + 2 * ball_1.mass * ball_1_parallel) / mass_sum
    
    
        ball_1.vel = ball_1_perpendicular + ball_1_parallel_final
        ball_2.vel = ball_2_perpendicular + ball_2_parallel_final
    
        ball_1.last_colision = frame * dt
        ball_2.last_colision = frame * dt


        # fixing overlap

        overlap = (ball_1.radius + ball_2.radius) - np.linalg.norm(ball_1.pos - ball_2.pos)
        correction = (overlap / 2) * collision_normal

        ball_1.pos = ball_1.pos - correction
        ball_2.pos = ball_2.pos + correction



    return 0



def calculate_collision_wall (ball, bounds):

    collision_normal = get_collision_normal_wall(ball, bounds)

    ball_parallel = project_vectors(ball.vel, collision_normal)
    ball.vel = ball.vel - 2 * ball_parallel



def integratePosVel(particle, deltaT=0.01):
    # utilizing the leapfrog method
    velHalf = particle.vel + particle.acel * deltaT / 2
    particle.pos = particle.pos + velHalf * deltaT
    #particle.acel = particle._getAcel(particle.pos, particles)
    particle.acel = 0
    particle.vel = velHalf + particle.acel * deltaT / 2



def get_rotation_matrix (alpha=0.0, beta=0.0, gamma=0.0, dtype='float64'):
    """
    alpha: angle of rotation around the x axis
    beta: angle of rotation around the y axis
    gamma: angle of rotation around the z axis
    """
    # It may be better to find a way to apply a rotation without using an external for loop.

    #rotation matrix in x
    rAlpha = np.array([[1,             0,              0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]], dtype=dtype)
    
    #rotation matrix in y
    rBeta = np.array([[np.cos(beta),  0,  np.sin(beta)],
                      [0,             1,             0],
                      [-np.sin(beta), 0, np.cos(beta)]], dtype=dtype)

    #rotation matrix in z
    rGamma = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                       [np.sin(gamma),  np.cos(gamma), 0],
                       [0,                          0, 1]], dtype=dtype)

    rGeneral = np.matmul(np.matmul(rGamma, rBeta), rAlpha, dtype=dtype)
    
    return rGeneral



def update_simulation_state (frame, particles, bounds, dt, ax):
    xs = []
    ys = []
    cs = []
    ss = []
    energy = 0.0
    deltaT = dt # Possible implementation of a variable timestep
    
    for n_i, i in enumerate(particles):
        integratePosVel(i, deltaT)
        energy += i.mass * np.linalg.norm(i.vel) ** 2 / 2
        for n_k, k in enumerate(particles[n_i:]):
            if i != k:
                if detect_collision(i, k):
                    #print("Collision!")
                    calculate_collision(i, k, frame, dt)
        
        if detect_collision_wall(i, bounds):
            calculate_collision_wall(i, bounds)

    ax.clear()
    
    circles = []
    for i in particles:
        circles.append(plt.Circle(i.pos[:2], radius=i.radius, linewidth=0,
                                  color=i.color))
    collection = matplotlib.collections.PatchCollection(circles)
    ax.add_collection(collection)

    energy = energy[0] # numpy.linalg.norm doesn't seem to return a scalar
    ax.annotate(f"{energy:0.2f}", (10, 10))

    plt.xlim(0, bounds)
    plt.ylim(0, bounds)



def generate_random_position (bounds):
    return np.random.random_integers(5, bounds - 5, 3)



def generate_random_velocity ():
    return np.random.random_integers(-10, 10, 3)



def generate_random_radius ():
    return np.random.random(1) * 5 + .5



def main ():

    DT = 1e-1 # Integration time step
    BOUNDS = 60 # Graph bounds (AU)
    ITVL_PYPLOT = 10 # Interval of each step in the animation (miliseconds)


    #rotation_matrix = get_rotation_matrix(0, 0, np.radians(0))


    #pos_1 = rotation_matrix @ (10 * np.array([0, 0, 0]))
    #pos_2 = rotation_matrix @ (10 * np.array([4, 0, 0]))
    #pos_3 = rotation_matrix @ (10 * np.array([2, 0, 0]))

    pos_1 = generate_random_position(BOUNDS)
    pos_2 = generate_random_position(BOUNDS)
    pos_3 = generate_random_position(BOUNDS)

    #vel_1 = rotation_matrix @ np.array([2, 2, 0])
    #vel_2 = rotation_matrix @ np.array([-2, 2, 0])
    #vel_3 = rotation_matrix @ np.array([0.01, 2, 0])

    vel_1 = generate_random_velocity()
    vel_2 = generate_random_velocity()
    vel_3 = generate_random_velocity()

    radius_1 = generate_random_radius()
    radius_2 = generate_random_radius()
    radius_3 = generate_random_radius()


    particles = [
        non_rotating_pool_ball(radius_1, radius_1 ** 2, pos_1, vel_1, color="red"),
        non_rotating_pool_ball(radius_2, radius_2 ** 2, pos_2, vel_2, color="blue"),
        non_rotating_pool_ball(radius_3, radius_3 ** 2, pos_3, vel_3, color="gold")
    ]


    fig = plt.figure()
    ax = plt.axes(xlim=(-BOUNDS, BOUNDS), ylim=(-BOUNDS, BOUNDS))
    ax.set_aspect('equal')

    simulate = lambda x: update_simulation_state(x, particles, BOUNDS, DT, ax)

    anim = animation.FuncAnimation(fig,
                                    simulate,
                                    interval=ITVL_PYPLOT,
                                    blit=False)

    plt.show()


if __name__ == "__main__":
    main()