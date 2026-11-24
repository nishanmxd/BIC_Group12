import numpy as np

"""
Algorithm 39: Particle Swarm Optimization (PSO)

1:  swarmsize ← desired swarm size
2:  α ← proportion of velocity to be retained
3:  β ← proportion of personal best to be retained
4:  γ ← proportion of the informants’ best to be retained
5:  δ ← proportion of global best to be retained
6:  ε ← jump size of a particle
7:  P ← {}
8:  for swarmsize times do
9:      P ← P union {new random particle x⃗ with a random initial velocity v⃗}
10: Best ← □
11: repeat
12:     for each particle x in P with velocity v do
13:         AssessFitness(x)
14:         if Best = □ or Fitness(x) > Fitness(Best) then
15:             Best ← x
16:     for each particle x in P with velocity v do
17:         x* ← previous fittest location of x
18:         x⁺ ← previous fittest location of informants of x
                 (including x itself)
19:         x! ← previous fittest location of any particle
20:         for each dimension i do
21:             b ← random number from 0.0 to β inclusive
22:             c ← random number from 0.0 to γ inclusive
23:             d ← random number from 0.0 to δ inclusive
24:             vᵢ ← αvᵢ + b(xᵢ* − xᵢ) + c(xᵢ⁺ − xᵢ) + d(xᵢ! − xᵢ)
25:     for each particle x⃗ ∈ P with velocity v⃗ do
26:         x ← x + εv
27: until Best is the ideal solution or we have run out of time
28: return Best
"""


class Particle:

    """
    Represents an individual particle in the swarm
    Each particle stores current position (ANN weights and biases), velocity, personal best position
    and personal best fitness
    """

    def __init__(self, dimensions, lower_bound, upper_bound):

        # Algorithm 39, Lines 8-9:
        # Create a new particle with a random initial velocity

        self.position = np.random.uniform(lower_bound, upper_bound, dimensions)

        # creates a random starting velocity
        self.velocity = np.random.uniform(-1, 1, dimensions)

        # store the starting position as the initial personal best
        self.best_position = self.position.copy()
        self.best_fitness = -np.inf


class PSO:

    """
    Implements Particle Swarm Optimisation using Algorithm 39
    from Essentials of Metaheuristics

    Uses personal best, informants best and global best
    to influence particle velocities
    """

    def __init__(
        self,
        fitness_function,
        dimensions,
        swarmsize=30,
        alpha=0.5,
        beta=1.0,
        gamma=1.0,
        delta=1.0,
        epsilon=0.1,
        informant_count=3,
        lower_bound=-1.0,
        upper_bound=1.0,
        boundary_method="reflect"
    ):

        # Algorithm 39, Lines 1-6:
        # Initialise PSO configuration parameters

        self.fitness_function = fitness_function
        self.dimensions = dimensions
        self.swarmsize = swarmsize
        
        self.alpha = alpha         #Alpha controls how much of the previous velocity is retained
        self.beta = beta           #Beta controls the influence of the particle's personal best
        self.gamma = gamma         #Gamma controls the influence of the informants' best position
        self.delta = delta         #Delta controls the influence of the overall global best position
        self.epsilon = epsilon     #Epsilon controls how far a particle moves using its velocity

        # Set the lower and upper limits for a particular positions
        self.lower_bound = np.broadcast_to(lower_bound, dimensions).copy()
        self.upper_bound = np.broadcast_to(upper_bound, dimensions).copy()

        # Check that the lower bounds are smaller than the upper bounds
        for i in range(self.dimensions):
            if self.lower_bound[i] >= self.upper_bound[i]:
                raise ValueError("Lower bounds must be less than upper bounds.")

        # Checks if the swarm contains at least two particles 
        if swarmsize < 2:
            raise ValueError("Swarm size must be at least 2.")

        # Checks the number of informant_count
        if informant_count < 1:
            raise ValueError("At least one informant is required.")

        #Only allow the two supported boundary handling methods
        if boundary_method not in ["clip", "reflect"]:
            raise ValueError("Boundary method must be 'clip' or 'reflect'.")

        #Check that all PSO parameters are zero or positive
        if any(value < 0 for value in [alpha, beta, gamma, delta, epsilon]):
            raise ValueError("PSO parameters must be non-negative.")

        # Limit the number of other informants to the available particles
        self.informant_count = min(informant_count, swarmsize - 1)
        self.boundary_method = boundary_method

        # Algorithm 39, Lines 7-10:
        # create an empty swarm and the global best

        self.swarm = []
        for i in range(swarmsize):
            particle = Particle(dimensions, self.lower_bound, self.upper_bound)
            self.swarm.append(particle)


        self.global_best_position = None
        self.global_best_fitness = -np.inf

        # Assign a random group of informants to each particle
        self.informants = self.assign_informants()

        # Store the best fitness after each iteration
        self.history = []

        self.position_history = []


    def assign_informants(self):

        """
        Randomly assigns a subset of other particles as
        informants for each particle

        Each particle is excluded from its own informant list
        """

        informants = []

        # Assign informants for each particle
        for i in range(self.swarmsize):
           
            possible_informants=[]
            for j in range (self.swarmsize):

                if j!=i:
                    possible_informants.append(j)        

            # Randomly select the required number of informants    
            selected = np.random.choice(possible_informants, size=self.informant_count, replace=False)                              #the same informant cannot be selected twice

            informants.append(selected)

        return informants

 

    def evaluate_fitness(self):

        """
        Evaluates every particle and updates its personal
        best and the global best

        Algorithm 39, Lines 12-16
        """

        # Evaluate each particle's current position
        for particle in self.swarm:

            fitness = self.fitness_function(particle.position)

            # Algorithm 39, Lines 13-15:
            # Compare current fitness with personal best fitness

            if fitness > particle.best_fitness:

                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            
            # Update global best if a better solution is found

            if fitness > self.global_best_fitness:

                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()



    def get_informant_best(self, particle_index):

        """
        Finds the personal best position belonging to the
        fittest informant of the specified particle.

        Algorithm 39, Line 18.
        """

        #Get the list of informants assigned to this particle
        informant_indices = self.informants[particle_index]

        #Assume this particle has the best fitness initially
        best_informant = particle_index


        # Compare the fitness of all informants and update if a better personal best is found
        for i in informant_indices:
            if self.swarm[i].best_fitness > self.swarm[best_informant].best_fitness:
                best_informant = i

        return self.swarm[best_informant].best_position.copy()



    def update_velocity(self):

        """
        Updates particle velocities using:

        1. Personal best.
        2. Informants' best.
        3. Global best.

        Algorithm 39, Lines 17-24.
        """

        # Update the velocity of each particle
        for i, particle in enumerate(self.swarm):

            # Algorithm 39, Line 17:
            # Retrieve the particle's previous fittest location

            personal_best = particle.best_position.copy()

            # Algorithm 39, Line 18:
            # Retrieve the previous fittest location of its informants

            informant_best = self.get_informant_best(i)

            # Algorithm 39, Line 19:
            # Retrieve the global best location

            global_best = self.global_best_position.copy()

            # Algorithm 39, Lines 21-23:
            # Generate random coefficients for each dimension

            b = np.random.uniform(0, self.beta, self.dimensions)
            c = np.random.uniform(0, self.gamma, self.dimensions)
            d = np.random.uniform(0, self.delta, self.dimensions)

            # Algorithm 39, Line 24:
            # Calculate the new velocity using all three values

            particle.velocity = (
                self.alpha * particle.velocity
                + b * (personal_best - particle.position)
                + c * (informant_best - particle.position)
                + d * (global_best - particle.position)
            )



    def handle_boundaries(self, particle):

        """
        Handles particles that move outside the permitted
        search space

        Extra beyond Algorithm 39

        Supports:
        - clip: Moves the particle to the nearest boundary
        - reflect: Reflects the particle back into the search space
        """

        if self.boundary_method == "clip":

            particle.position = np.clip(particle.position, self.lower_bound, self.upper_bound)

            # Stop outward movement at the boundary
            particle.velocity = np.where( (particle.position <= self.lower_bound) | (particle.position >= self.upper_bound),
                                          0,
                                          particle.velocity
                                         )

        elif self.boundary_method == "reflect":

            width = self.upper_bound - self.lower_bound

            distance = particle.position - self.lower_bound

            reflected_distance = np.mod(distance, 2 * width)

            # Map the particle back into the search space
            particle.position = self.lower_bound + np.where(reflected_distance <= width, reflected_distance,
                                                            2 * width - reflected_distance
                                                            )

            # Reverse velocity when reflection occurs
            particle.velocity = np.where(reflected_distance > width, -particle.velocity, particle.velocity)



    def update_position(self):

        """
        Moves each particle according to its updated velocity

        Algorithm 39, Lines 25-26
        """
        # Move every particle according to its velocity and jump size
        for particle in self.swarm:

            particle.position += self.epsilon * particle.velocity

            self.handle_boundaries(particle)


    def optimize(self, max_iterations=100, target_fitness=None):

        """
        Runs the PSO algorithm until the maximum number of
        iterations is reached or the target fitness is achieved

        Algorithm 39, Lines 11-28
        """

        for iteration in range(max_iterations):

            # Algorithm 39, Lines 12-15:
            # Evaluate fitness and update personal/global bests

            self.evaluate_fitness()

            # Store the best fitness for monitoring progress
            self.history.append(self.global_best_fitness)

            self.position_history.append(self.global_best_position.copy())

            print(
                f"Iteration {iteration + 1}/{max_iterations} | "
                f"Best Fitness: {self.global_best_fitness:.6f}"
            )

            # Algorithm 39, Line 27:
            # Stop if the ideal solution has been reached

            
            if (target_fitness is not None and self.global_best_fitness >= target_fitness):
                break
               
            if iteration == max_iterations - 1:
                break

            # Algorithm 39, Lines 16-24:
            # Calculate new velocities for all particles

            self.update_velocity()

            # Algorithm 39, Lines 25-26:
            # Update particle positions

            self.update_position()

        

        # Algorithm 39, Line 28:
        # Return the global best solution

        return self.global_best_position, self.global_best_fitness, self.history