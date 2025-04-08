#include <assert.h>
#include <cinttypes>
#include <cstdio>

typedef uint64_t UID;

struct Position {
    float p[3];
};

struct Velocity {
    float v[3];
};

struct Force {
    float f[3];
};

struct Agent {
    UID id;
    Position position;
    Velocity velocity;
};

struct AgentContainer {
    Agent agent;
};

struct Runner {
    /// Counter to assign unique identifiers to new agents
    uint64_t uid_counter;

    /// Number of agents currently stored
    uint64_t n_agents;

    /// Values of the Agents are fully updated.
    bool agents_up_to_date;

    // This is a pointer to a single element which marks the first element of an array.
    // The length of this array is given by the previous value.
    AgentContainer *agents;

    /// Pointer to an array of floats which stores the positions of the agents
    float *positions;

    /// Pointer to an array of floats which stores the velocities of the agents
    float *velocities;

    /// Pointer to an array of floats which stores the forces of the agents
    float *forces;

    /// Function pointer which calculates the interaction force between two agents.
    // Force (*fptr)(Position, Velocity, Position, Velocity);
};

__global__ void calculate_forces(int total, float *pos1, float *vel1, float *pos2, float *vel2,
                                 float *forces) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= total || j >= total) {
        return;
    }

    float p1, p2;
    // float v1, v2;
    p1 = pos1[i];
    // v1 = vel1[i];
    p2 = pos2[j];
    // v2 = vel2[j];

    forces[i] += 0.1 * (p1 - p2);
    forces[j] += 0.1 * (p2 - p1);
}

extern "C" Runner *new_runner(Agent *agents, uint64_t n_agents) {

    // float positions[3 * n_agents] = malloc(3 * n_agents * sizeof(float));
    float *positions, *velocities, *forces;
    positions  = (float *) malloc(n_agents * sizeof(float));
    velocities = (float *) malloc(n_agents * sizeof(float));
    forces     = (float *) malloc(n_agents * sizeof(float));

    AgentContainer *agent_containers;
    agent_containers = (AgentContainer *) malloc(n_agents * sizeof(AgentContainer));
    for (uint64_t k = 0; k < n_agents; k++) {
        agent_containers[k] = AgentContainer{
            .agent = agents[k],
        };
    }
    struct Runner runner{
        n_agents, n_agents, true, agent_containers, positions, velocities, forces,
    };
    struct Runner *runner_ptr = &runner;
    return runner_ptr;
}

extern "C" void drop_runner(Runner *runner) {
    free(runner->agents);
    free(runner->positions);
    free(runner->velocities);
    free(runner->forces);
}

/// This calculates with time complexity O(n^2) which is not ideal but the most simple thing that
// one can write down so we will stick with it for now.
extern "C" void update_positions(Runner &runner) {
    for (uint64_t i = 0; i < runner.n_agents; i++) {
        for (uint64_t j = 0; j < runner.n_agents; j++) {
            float p1[3] = {runner.positions[i], runner.positions[i + 1], runner.positions[i + 2]};
            /* float v1[3] = {runner.velocities[i], runner.velocities[i + 1],
                           runner.velocities[i + 2]};*/
            float p2[3] = {runner.positions[j], runner.positions[j + 1], runner.positions[j + 2]};
            /* float v2[3] = {runner.velocities[j], runner.velocities[j + 1],
                           runner.velocities[j + 2]};*/

            float f[3] = {0.0};
            for (uint8_t k = 0; k < 3; k++) {
                // Calculate the force between agents
                f[k] += 0.1 * (p1 - p2);

                // Update forces
                // runner.forces[i + k] += f[k];
                // runner.forces[j + k] -= f[k];
            }
        }
    }
    return;
}

extern "C" void print_positions(Runner &runner) {
    for (uint64_t i = 0; i < runner.n_agents; i++) {
        Agent agent = runner.agents[i].agent;
        Position p  = agent.position;
        Velocity v  = agent.velocity;
        // printf("%lu p=[%f, %f, %f] v=[%f, %f, %f]\n", agent.id, p.p[0], p.p[1], p.p[2], v.v[0],
        //        v.v[1], v.v[2]);
    }
}

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

/// This is a comment
extern "C" void do_compute() {
    int N = 1 << 20;
    float *x, *y, *d_x, *d_y;
    x = (float *) malloc(N * sizeof(float));
    y = (float *) malloc(N * sizeof(float));

    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = abs(y[i] - 4.0f);
        // printf("%f\n", maxError);
        // assert(maxError < 0.001);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
