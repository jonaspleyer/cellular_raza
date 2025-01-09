#include <assert.h>
#include <cinttypes>
#include <cstdio>

typedef uint32_t UID;

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
    Force current_force;
};

struct Runner {
    uint32_t n_agents;
    // This is a pointer to a single element which marks the first element of an array.
    // The length of this array is given by the previous value.
    Agent *agents;
};

extern "C" void update_positions(Runner &runner) {
    for (uint32_t i = 0; i < runner.n_agents; i++) {
        for (uint32_t j = 0; j < runner.n_agents; j++) {
        }
    }
}

extern "C" void print_positions(Runner &runner) {
    printf("%p\n\n\n", runner.agents);
    for (uint32_t i = 0; i < runner.n_agents; i++) {
        Agent agent = runner.agents[i];
        Position p  = agent.position;
        Velocity v  = agent.velocity;
        printf("%u p=[%f, %f, %f] v=[%f, %f, %f]\n", agent.id, p.p[0], p.p[1], p.p[2], v.v[0],
               v.v[1], v.v[2]);
    }
}

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

/// This is a comment
extern "C" void do_compute(void) {
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
        maxError = max(maxError, abs(y[i] - 4.0f));
        assert(maxError < 0.001);
    }

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
