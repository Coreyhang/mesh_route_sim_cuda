#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#define DEPTH 5
#define DOUBLE_DEPTH 10
#define CHANNEL_NUM 4

// #define DEBUG

#ifdef DEBUG
#define debug(format, ...) printf(format, ##__VA_ARGS__)
#else
#define debug(format, ...)
#endif

int router_busy = 0;
int router_empty = 1;

// one route packet
struct Packet
{
    int dstx;
    int dsty;
    int num;
};

struct ChannelOut
{
    struct Packet *out;
    int delta_credit;
};

struct ChannelOut *channel_out;
struct Packet *all_packets;

struct Router
{
    int x;
    int y;
    int id;
    // local
    struct Packet *local;
    int local_size;
    int local_start;
    struct Packet *local_out; // virtual output buffer of local
    // left right up down
    struct Packet *channels[CHANNEL_NUM][DEPTH];
    int channels_size[CHANNEL_NUM];
    int channels_start[CHANNEL_NUM];
    int channels_end[CHANNEL_NUM];
    int channels_credit[CHANNEL_NUM];
    struct ChannelOut *channels_connected[CHANNEL_NUM];
};

__device__ inline void arbit_one_channel(struct Router *d_r, struct ChannelOut *d_channel_out, int *d_router_empty, int *d_router_busy, struct Packet *p, int i)
{
    if (d_r->channels_size[i] > 0)
    { // left channel's queue not empty
        *d_router_empty = 0;
        p = d_r->channels[i][d_r->channels_start[i]];
        if (p->dstx > d_r->x)
        {
            if (d_channel_out[(d_r->id) * CHANNEL_NUM + 1].out == NULL && d_r->channels_credit[1] > 0)
            {
                d_channel_out[(d_r->id) * CHANNEL_NUM + 1].out = p; // assign to ChannelOut
                d_r->channels_credit[1] -= 1;
                d_r->channels_start[i] = (d_r->channels_start[i] + 1) % DEPTH; // pop first packet
                d_r->channels_size[i] -= 1;
                d_channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                *d_router_busy = 1;
                debug("\t\t channel %d -> right\n", i);
            }
            // else
            //     channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
        }
        else if (p->dstx < d_r->x)
        {
            if (d_channel_out[(d_r->id) * CHANNEL_NUM + 0].out == NULL && d_r->channels_credit[0] > 0)
            {
                d_channel_out[(d_r->id) * CHANNEL_NUM + 0].out = p; // assign to ChannelOut
                d_r->channels_credit[0] -= 1;
                d_r->channels_start[i] = (d_r->channels_start[i] + 1) % DEPTH; // pop first packet
                d_r->channels_size[i] -= 1;
                d_channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                *d_router_busy = 1;
                debug("\t\t channel %d -> left\n", i);
            }
            // else
            //     channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
        }
        else if (p->dsty > d_r->y)
        {
            if (d_channel_out[(d_r->id) * CHANNEL_NUM + 2].out == NULL && d_r->channels_credit[2] > 0)
            {
                d_channel_out[(d_r->id) * CHANNEL_NUM + 2].out = p; // assign to ChannelOut
                d_r->channels_credit[2] -= 1;
                d_r->channels_start[i] = (d_r->channels_start[i] + 1) % DEPTH; // pop first packet
                d_r->channels_size[i] -= 1;
                d_channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                *d_router_busy = 1;
                debug("\t\t channel %d -> up\n", i);
            }
            // else
            //     channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
        }
        else if (p->dsty < d_r->y)
        {
            if (d_channel_out[(d_r->id) * CHANNEL_NUM + 3].out == NULL && d_r->channels_credit[3] > 0)
            {
                d_channel_out[(d_r->id) * CHANNEL_NUM + 3].out = p; // assign to ChannelOut
                d_r->channels_credit[3] -= 1;
                d_r->channels_start[i] = (d_r->channels_start[i] + 1) % DEPTH; // pop first packet
                d_r->channels_size[i] -= 1;
                d_channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                *d_router_busy = 1;
                debug("\t\t channel %d -> down\n", i);
            }
            // else
            //     channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
        }
        else if (d_r->local_out == NULL)
        {
            d_r->local_out = p;
            d_r->channels_start[i] = (d_r->channels_start[i] + 1) % DEPTH; // pop first packet
            d_r->channels_size[i] -= 1;
            d_channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
            *d_router_busy = 1;
            debug("\t\t channel %d -> local\n", i);
        }
        else
        {
            d_channel_out[(d_r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
        }
    }
}

__device__ void router_arbit(struct Router *d_r, struct ChannelOut *d_channel_out, int *d_router_empty, int *d_router_busy)
{
    struct Packet *p = NULL;
    arbit_one_channel(d_r, d_channel_out, d_router_empty, d_router_busy, p, 0);
    arbit_one_channel(d_r, d_channel_out, d_router_empty, d_router_busy, p, 1);
    arbit_one_channel(d_r, d_channel_out, d_router_empty, d_router_busy, p, 2);
    arbit_one_channel(d_r, d_channel_out, d_router_empty, d_router_busy, p, 3);
    if (d_r->local_size > 0)
    { // local channel's queue not empty
        *d_router_empty = 0;
        p = &d_r->local[d_r->local_start];
        if (p->dstx > d_r->x)
        {
            if (d_channel_out[(d_r->id) * CHANNEL_NUM + 1].out == NULL && d_r->channels_credit[1] > 0)
            {
                d_channel_out[(d_r->id) * CHANNEL_NUM + 1].out = p; // assign to ChannelOut
                d_r->channels_credit[1] -= 1;
                if (p->num == 1)
                {
                    d_r->local_start += 1; // pop first packet
                    d_r->local_size -= 1;
                }
                else
                    p->num -= 1;
                *d_router_busy = 1;
                debug("\t\t local -> right\n");
            }
        }
        else if (p->dstx < d_r->x)
        {
            if (d_channel_out[(d_r->id) * CHANNEL_NUM + 0].out == NULL && d_r->channels_credit[0] > 0)
            {
                d_channel_out[(d_r->id) * CHANNEL_NUM + 0].out = p; // assign to ChannelOut
                d_r->channels_credit[0] -= 1;
                if (p->num == 1)
                {
                    d_r->local_start += 1; // pop first packet
                    d_r->local_size -= 1;
                }
                else
                    p->num -= 1;
                *d_router_busy = 1;
                debug("\t\t local -> left\n");
            }
        }
        else if (p->dsty > d_r->y)
        {
            if (d_channel_out[(d_r->id) * CHANNEL_NUM + 2].out == NULL && d_r->channels_credit[2] > 0)
            {
                d_channel_out[(d_r->id) * CHANNEL_NUM + 2].out = p; // assign to ChannelOut
                d_r->channels_credit[2] -= 1;
                if (p->num == 1)
                {
                    d_r->local_start += 1; // pop first packet
                    d_r->local_size -= 1;
                }
                else
                    p->num -= 1;
                *d_router_busy = 1;
                debug("\t\t local -> up\n");
            }
        }
        else if (p->dsty < d_r->y)
        {
            if (d_channel_out[(d_r->id) * CHANNEL_NUM + 3].out == NULL && d_r->channels_credit[3] > 0)
            {
                d_channel_out[(d_r->id) * CHANNEL_NUM + 3].out = p; // assign to ChannelOut
                d_r->channels_credit[3] -= 1;
                if (p->num == 1)
                {
                    d_r->local_start += 1; // pop first packet
                    d_r->local_size -= 1;
                }
                else
                    p->num -= 1;
                *d_router_busy = 1;
                debug("\t\t local -> down\n");
            }
        }
        else if (d_r->local_out == NULL)
        {
            d_r->local_out = p;
            if (p->num == 1)
            {
                d_r->local_start += 1; // pop first packet
                d_r->local_size -= 1;
            }
            else
                p->num -= 1;
            *d_router_busy = 1;
            debug("\t\t local -> local\n");
        }
    }
}

__device__ void router_trans(struct Router *d_r)
{
    // for (int i = 0; i < CHANNEL_NUM; i++)
    // {

    // }
    if (d_r->channels_connected[0] != NULL)
    {
        d_r->channels_credit[0] += d_r->channels_connected[0]->delta_credit; // update credit
        d_r->channels_connected[0]->delta_credit = 0;
        if (d_r->channels_connected[0]->out != NULL)
        {
            d_r->channels[0][d_r->channels_end[0]] = d_r->channels_connected[0]->out; // push packet in input buffer
            d_r->channels_end[0] = (d_r->channels_end[0] + 1) % DEPTH;
            d_r->channels_size[0] += 1;
            d_r->channels_connected[0]->out = NULL;
            debug("\t\t channel %d get\n", 0); // clear connected channel's out sign, prepare for next arbit
        }
    }
    if (d_r->channels_connected[1] != NULL)
    {
        d_r->channels_credit[1] += d_r->channels_connected[1]->delta_credit; // update credit
        d_r->channels_connected[1]->delta_credit = 0;
        if (d_r->channels_connected[1]->out != NULL)
        {
            d_r->channels[1][d_r->channels_end[1]] = d_r->channels_connected[1]->out; // push packet in input buffer
            d_r->channels_end[1] = (d_r->channels_end[1] + 1) % DEPTH;
            d_r->channels_size[1] += 1;
            d_r->channels_connected[1]->out = NULL;
            debug("\t\t channel %d get\n", 1); // clear connected channel's out sign, prepare for next arbit
        }
    }
    if (d_r->channels_connected[2] != NULL)
    {
        d_r->channels_credit[2] += d_r->channels_connected[2]->delta_credit; // update credit
        d_r->channels_connected[2]->delta_credit = 0;
        if (d_r->channels_connected[2]->out != NULL)
        {
            d_r->channels[2][d_r->channels_end[2]] = d_r->channels_connected[2]->out; // push packet in input buffer
            d_r->channels_end[2] = (d_r->channels_end[2] + 1) % DEPTH;
            d_r->channels_size[2] += 1;
            d_r->channels_connected[2]->out = NULL;
            debug("\t\t channel %d get\n", 2); // clear connected channel's out sign, prepare for next arbit
        }
    }
    if (d_r->channels_connected[3] != NULL)
    {
        d_r->channels_credit[3] += d_r->channels_connected[3]->delta_credit; // update credit
        d_r->channels_connected[3]->delta_credit = 0;
        if (d_r->channels_connected[3]->out != NULL)
        {
            d_r->channels[3][d_r->channels_end[3]] = d_r->channels_connected[3]->out; // push packet in input buffer
            d_r->channels_end[3] = (d_r->channels_end[3] + 1) % DEPTH;
            d_r->channels_size[3] += 1;
            d_r->channels_connected[3]->out = NULL;
            debug("\t\t channel %d get\n", 3); // clear connected channel's out sign, prepare for next arbit
        }
    }
    d_r->local_out = NULL; // clear local output buffer's out sign
}

__global__ void clear(int *d_router_empty, int *d_router_busy)
{
    *d_router_empty = 1;
    *d_router_busy = 0;
}

__global__ void g_router_arbit(struct Router *d_router_list, struct ChannelOut *d_channel_out, int *d_router_empty, int *d_router_busy, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadIdx.x == 0)
    {
        *d_router_empty = 1;
        *d_router_busy = 0;
    }
    __syncthreads();
    while (tid < N)
    {
        debug("tid %d; empty %d, busy %d\n", tid, *d_router_empty, *d_router_busy);
        router_arbit(&d_router_list[tid], d_channel_out, d_router_empty, d_router_busy);
        tid += gridDim.x * blockDim.x;
    }
}

__global__ void g_router_trans(struct Router *d_router_list, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    while (tid < N)
    {
        debug("tid %d; trans\n", tid);
        router_trans(&d_router_list[tid]);
        tid += gridDim.x * blockDim.x;
    }
}

// random generate route table
int rand_init_route_table(int **rt, int dim_x, int dim_y, int connect_num, int max_send_num)
{
    int total_packets_num = 0;
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        // for (int m = 0; m < connect_num; m++)
        // {
        //     rt[i][rand() % (dim_x * dim_y)] = rand() % max_send_num + 1;
        //     total_packets_num += 1;
        // }
        for (int j = 0; j < dim_x * dim_y; j++)
        {
            if ((rand() % 100000) < connect_num)
            {
                rt[i][j] = rand() % max_send_num + 1;
                total_packets_num += 1;
            }
        }
    }
    return total_packets_num;
}

void print_route_table(int **rt, int dim_x, int dim_y)
{
    printf("xx\t");
    for (int j = 0; j < dim_x * dim_y; j++)
    {
        printf("%d\t", j);
    }
    printf("\n");
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        printf("%d\t", i);
        for (int j = 0; j < dim_x * dim_y; j++)
        {
            printf("%d\t", rt[i][j]);
        }
        printf("\n");
    }
}

void init_topo_mesh(struct Router *rl, int dim_x, int dim_y)
{
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        rl[i].x = i % dim_x;
        rl[i].y = i / dim_x;
        rl[i].id = i;
        for (int j = 0; j < CHANNEL_NUM; j++)
            rl[i].channels_credit[j] = DEPTH;
    }

    // init connection
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        int x = i % dim_x;
        int y = i / dim_x;
        // left right up down
        rl[i].channels_connected[0] = (x == 0) ? NULL : &channel_out[(x - 1 + y * dim_x) * CHANNEL_NUM + 1];
        rl[i].channels_connected[1] = (x == dim_x - 1) ? NULL : &channel_out[(x + 1 + y * dim_x) * CHANNEL_NUM + 0];
        rl[i].channels_connected[2] = (y == dim_y - 1) ? NULL : &channel_out[(x + (y + 1) * dim_x) * CHANNEL_NUM + 3];
        rl[i].channels_connected[3] = (y == 0) ? NULL : &channel_out[(x + (y - 1) * dim_x) * CHANNEL_NUM + 2];
    }
}

void init_packets(struct Router *rl, int **rt, int dim_x, int dim_y, int total_packets_num)
{
    int packet_ptr = 0;
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        rl[i].local = &all_packets[packet_ptr];
        rl[i].local_size = 0;
        // local
        for (int j = 0; j < dim_x * dim_y; j++)
        {
            if (rt[i][j] > 0)
            {
                all_packets[packet_ptr].dstx = j % dim_x;
                all_packets[packet_ptr].dsty = j / dim_x;
                all_packets[packet_ptr].num = rt[i][j];
                rl[i].local_size += 1;
                packet_ptr += 1;
            }
        }
    }
    assert(packet_ptr == total_packets_num);
}

__global__ void sim(struct Router *d_router_list, struct ChannelOut *d_channel_out, int *d_router_empty, int *d_router_busy, int N)
{
    int tid;
    int cycle = 0;
    while (1)
    {
        debug("cycle %d\n", cycle);
        *d_router_empty = 1;
        *d_router_busy = 0;
        tid = blockDim.x * blockIdx.x + threadIdx.x;
        while (tid < N)
        {
            debug("tid %d; empty %d, busy %d\n", tid, *d_router_empty, *d_router_busy);
            router_arbit(&d_router_list[tid], d_channel_out, d_router_empty, d_router_busy);
            tid += gridDim.x * blockDim.x;
            // debug("\t\t; empty %d, busy %d\n", *d_router_empty, *d_router_busy);
        }
        __syncthreads();
        tid = blockDim.x * blockIdx.x + threadIdx.x;
        while (tid < N)
        {
            debug("tid %d; trans\n", tid);
            router_trans(&d_router_list[tid]);
            tid += gridDim.x * blockDim.x;
        }
        __syncthreads();
        debug("* tid %d; empty %d, busy %d\n", blockDim.x * blockIdx.x + threadIdx.x, *d_router_empty, *d_router_busy);
        if (*d_router_busy == 0)
        {
            printf("cycle %d\n", cycle);
            return;
        }
        cycle++;
    }
    debug("%d SIM DONE!\n", blockDim.x * blockIdx.x + threadIdx.x);
}

int main()
{
    time_t t;
    // srand((unsigned)time(&t));
    srand(1);

    struct timeval time_start, time_end;

    int dim_x = 1000;
    int dim_y = 100;

    int p_send = 10; // 1/100,000
    int max_send_num = 1000;

    int query = 1000;

    int total_packets_num = 0;
    int **route_table = (int **)malloc(sizeof(int *) * dim_x * dim_y);
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        route_table[i] = (int *)calloc(dim_x * dim_y, sizeof(int));
    }

    total_packets_num = rand_init_route_table(route_table, dim_x, dim_y, p_send, max_send_num);
    // route_table[0][1] = 10;
    // // route_table[0][3] = 10;
    // route_table[1][0] = 10;
    // total_packets_num = 2;

    // print_route_table(route_table, dim_x, dim_y);

    channel_out = (struct ChannelOut *)calloc(dim_x * dim_y * CHANNEL_NUM, sizeof(struct ChannelOut));
    all_packets = (struct Packet *)malloc(total_packets_num * sizeof(struct Packet));
    struct Router *router_list = (struct Router *)calloc(dim_x * dim_y, sizeof(struct Router));

    init_topo_mesh(router_list, dim_x, dim_y);

    // init packets
    init_packets(router_list, route_table, dim_x, dim_y, total_packets_num);

    int cycle = 0;

    // CUDA malloc
    struct ChannelOut *d_channel_out;
    struct Packet *d_all_packets;
    struct Router *d_router_list;
    int *d_cycle, *d_empty, *d_busy;
    cudaMalloc((void **)&d_channel_out, dim_x * dim_y * CHANNEL_NUM * sizeof(struct ChannelOut));
    cudaMalloc((void **)&d_all_packets, total_packets_num * sizeof(struct Packet));
    cudaMalloc((void **)&d_router_list, dim_x * dim_y * sizeof(struct Router));
    cudaMalloc((void **)&d_cycle, sizeof(int));
    cudaMalloc((void **)&d_empty, sizeof(int));
    cudaMalloc((void **)&d_busy, sizeof(int));

    //
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        if (router_list[i].local != NULL)
            router_list[i].local = d_all_packets + (router_list[i].local - all_packets);
        for (int j = 0; j < CHANNEL_NUM; j++)
        {
            if (router_list[i].channels_connected[j] != NULL)
                router_list[i].channels_connected[j] = d_channel_out + (router_list[i].channels_connected[j] - channel_out);
        }
    }

    cudaMemcpy(d_channel_out, channel_out, dim_x * dim_y * CHANNEL_NUM * sizeof(struct ChannelOut), cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_packets, all_packets, total_packets_num * sizeof(struct Packet), cudaMemcpyHostToDevice);
    cudaMemcpy(d_router_list, router_list, dim_x * dim_y * sizeof(struct Router), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cycle, &cycle, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_empty, &router_empty, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_busy, &router_busy, sizeof(int), cudaMemcpyHostToDevice);

    // gettimeofday(&time_start, NULL);
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // start sim
    int dead_lock_cnt = 0;
    printf("Start ...\n");

    cudaEventRecord(start, 0);

    // sim<<<2, 1>>>(d_router_list, d_channel_out, d_empty, d_busy, dim_x * dim_y);
    // int cycle = 0;
    while (1)
    {
        debug("cycle %d\n", cycle);
        // *d_router_empty = 1;
        // *d_router_busy = 0;

        // clear<<<1, 1>>>(d_empty, d_busy);

        g_router_arbit<<<4096, 128>>>(d_router_list, d_channel_out, d_empty, d_busy, dim_x * dim_y);

        g_router_trans<<<4096, 128>>>(d_router_list, dim_x * dim_y);

        // cudaMemcpy(&router_busy, d_busy, sizeof(int), cudaMemcpyDeviceToHost);

        if (cycle % query == 0)
        {
            cudaMemcpy(&router_busy, d_busy, sizeof(int), cudaMemcpyDeviceToHost);
            if (router_busy == 0)
            {
                printf("End, cycle %d \n", cycle);
                break;
            }
        }
        cycle++;
    }

    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, end);
    printf("CUDA time = %f ms\n", elapsedTime);

    // gettimeofday(&time_end, NULL);
    // double total_time = ((double)(time_end.tv_sec) + (double)(time_end.tv_usec) / 1000000.0) - ((double)(time_start.tv_sec) + (double)(time_start.tv_usec) / 1000000.0);
    // printf("Using time = %f s\n", total_time);

    cudaMemcpy(&cycle, d_cycle, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&router_empty, d_empty, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&router_busy, d_busy, sizeof(int), cudaMemcpyDeviceToHost);

    printf("**** %d, %d, %d\n", cycle, router_empty, router_busy);

    return 0;
}