#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <pthread.h>

#define DEPTH 5
#define CHANNEL_NUM 4

#define THREAD_NUM 25

// #define DEBUG

#ifdef DEBUG
#define debug(format, ...) printf(format, ##__VA_ARGS__)
#else
#define debug(format, ...)
#endif

// int router_busy = 0;
// int router_empty = 1;

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

pthread_cond_t p_all_sync1 = PTHREAD_COND_INITIALIZER;
pthread_mutex_t p_lock1 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t p_all_sync2 = PTHREAD_COND_INITIALIZER;
pthread_mutex_t p_lock2 = PTHREAD_MUTEX_INITIALIZER;

int all_sync1, all_sync2;

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

struct ThreadParam
{
    int pid;
    int size;
    struct Router *router_list;
    struct ChannelOut *channel_out;
    int *router_empty;
    int *router_busy;
    int *sync;
    int dim_x;
    int dim_y;
};

void arbit_one_channel(struct Router *d_r, struct ChannelOut *d_channel_out, int *d_router_empty, int *d_router_busy, struct Packet *p, int i)
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

void router_arbit(struct Router *d_r, struct ChannelOut *d_channel_out, int *d_router_empty, int *d_router_busy)
{
    struct Packet *p = NULL;
    // for (int i = 0; i < CHANNEL_NUM; i++)
    // {
    // }
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

void router_trans(struct Router *d_r)
{
    for (int i = 0; i < CHANNEL_NUM; i++)
    {
        if (d_r->channels_connected[i] != NULL)
        {
            d_r->channels_credit[i] += d_r->channels_connected[i]->delta_credit; // update credit
            d_r->channels_connected[i]->delta_credit = 0;
            if (d_r->channels_connected[i]->out != NULL)
            {
                d_r->channels[i][d_r->channels_end[i]] = d_r->channels_connected[i]->out; // push packet in input buffer
                d_r->channels_end[i] = (d_r->channels_end[i] + 1) % DEPTH;
                d_r->channels_size[i] += 1;
                d_r->channels_connected[i]->out = NULL;
                debug("\t\t channel %d get\n", i); // clear connected channel's out sign, prepare for next arbit
            }
        }
    }
    d_r->local_out = NULL; // clear local output buffer's out sign
}

void one_thread(void *arg)
{
    struct ThreadParam *tp;
    tp = (struct ThreadParam *)arg;
    int pid = tp->pid;
    int size = tp->size;
    struct Router *router_list = tp->router_list;
    struct ChannelOut *channel_out = tp->channel_out;
    int *router_empty = tp->router_empty;
    int *router_busy = tp->router_busy;
    int *sync = tp->sync;
    int dim_x = tp->dim_x;
    int dim_y = tp->dim_y;
    int cycle = 0;
    int any_busy = 0;
    printf("*** one thread start : %d\n", pid);
    while (1)
    {
        // arbit
        for (int i = pid * size; i < (pid + 1) * size; i++)
        {
            debug("-- cycle %d : thread %d : - arbit (%d, %d)\n", cycle, pid, i % dim_x, i / dim_x);
            router_arbit(&router_list[i], channel_out, &router_empty[pid], &router_busy[pid]);
        }
        // **************** sync ****************
        // pthread_mutex_lock(&p_lock1);
        // sync[pid] = 1;
        // debug("-- cycle %d : thread %d :cond wait 1 \n", cycle, pid);
        // pthread_cond_wait(&p_all_sync1, &p_lock1);
        // pthread_mutex_unlock(&p_lock1);
        // debug("-- cycle %d : thread %d :cond trigger 1 \n", cycle, pid);
        // sync[pid] = 0;

        sync[pid] = 1;
        debug("-- cycle %d : thread %d :cond wait 1 \n", cycle, pid);
        while (all_sync1 == 0)
        {
        };
        debug("-- cycle %d : thread %d :cond trigger 1 \n", cycle, pid);
        sync[pid] = 0;

        // transmit
        for (int i = pid * size; i < (pid + 1) * size; i++)
        {
            debug("-- cycle %d : thread %d : - trans (%d, %d)\n", cycle, pid, i % dim_x, i / dim_x);
            router_trans(&router_list[i]);
        }
        // **************** sync ****************
        // pthread_mutex_lock(&p_lock2);
        // sync[pid + THREAD_NUM] = 1;
        // debug("-- cycle %d : thread %d :cond wait 2 \n", cycle, pid);
        // pthread_cond_wait(&p_all_sync2, &p_lock2);
        // pthread_mutex_unlock(&p_lock2);
        // debug("-- cycle %d : thread %d :cond trigger 2 \n", cycle, pid);
        // sync[pid + THREAD_NUM] = 0;

        sync[pid + THREAD_NUM] = 1;
        debug("-- cycle %d : thread %d :cond wait 2 \n", cycle, pid);
        while (all_sync2 == 0)
        {
        };
        debug("-- cycle %d : thread %d :cond trigger 2 \n", cycle, pid);
        sync[pid + THREAD_NUM] = 0;

        router_busy[pid] = 0;

        cycle += 1;
        debug("-- cycle %d : thread %d : %d cycle end ----- \n", cycle, pid, cycle);
    }
    printf("*** one thread end\n");
}

void main_thread(void *arg)
{
    printf("*** main thread start\n");
    struct ThreadParam *tp;
    tp = (struct ThreadParam *)arg;
    int pid = tp->pid;
    int size = tp->size;
    struct Router *router_list = tp->router_list;
    struct ChannelOut *channel_out = tp->channel_out;
    int *router_empty = tp->router_empty;
    int *router_busy = tp->router_busy;
    int *sync = tp->sync;
    int dim_x = tp->dim_x;
    int dim_y = tp->dim_y;
    int all_sync = 0;
    all_sync1 = 0;
    all_sync2 = 0;
    int cycle = 0;
    int any_busy = 0;
    while (1)
    {
        // arbit
        for (int i = pid * size; i < (pid + 1) * size; i++)
        {
            debug("-- cycle %d : thread %d : - arbit (%d, %d)\n", cycle, pid, i % dim_x, i / dim_x);
            router_arbit(&router_list[i], channel_out, &router_empty[pid], &router_busy[pid]);
        }
        // **************** sync ****************
        sync[pid] = 1;
        while (1)
        {
            all_sync = sync[0];
            for (int i = 1; i < THREAD_NUM; i++)
            {
                all_sync &= sync[i];
            }
            all_sync1 = all_sync;
            if (all_sync1 == 1)
            {
                debug("-- cycle %d : thread %d :cond broadcast 1 \n", cycle, pid);
                break;
            }
        }
        sync[pid] = 0;
        while (1)
        {
            all_sync = sync[0];
            for (int i = 1; i < THREAD_NUM; i++)
            {
                all_sync |= sync[i];
            }
            if (all_sync == 0)
            {
                all_sync1 = 0;
                break;
            }
        }

        // End condition
        // printf("-- thread %d : cycle : %d\n", pid, cycle);
        any_busy = router_busy[0];
        for (int i = 1; i < THREAD_NUM; i++)
        {
            any_busy |= router_busy[i];
        }
        if (any_busy == 0)
        {
            printf("pid %d Total cycle : %d\n", pid, cycle);
            printf("pid %d End ...\n", pid);
            break;
        }
        // transmit
        for (int i = pid * size; i < (pid + 1) * size; i++)
        {
            debug("-- cycle %d : thread %d : - trans (%d, %d)\n", cycle, pid, i % dim_x, i / dim_x);
            router_trans(&router_list[i]);
        }
        // **************** sync ****************
        sync[pid + THREAD_NUM] = 1;
        while (1)
        {
            all_sync = sync[0 + THREAD_NUM];
            for (int i = 1; i < THREAD_NUM; i++)
            {
                all_sync &= sync[i + THREAD_NUM];
            }
            all_sync2 = all_sync;
            if (all_sync2 == 1)
            {
                debug("-- cycle %d : thread %d :cond broadcast 2 \n", cycle, pid);
                break;
            }
        }
        sync[pid + THREAD_NUM] = 0;
        while (1)
        {
            all_sync = sync[0 + THREAD_NUM];
            for (int i = 1; i < THREAD_NUM; i++)
            {
                all_sync |= sync[i + THREAD_NUM];
            }
            if (all_sync == 0)
            {
                all_sync2 = 0;
                break;
            }
        }

        router_busy[pid] = 0;

        cycle += 1;
        debug("--cycle %d : thread %d : %d cycle end ----- \n", cycle, pid, cycle);
    }
    printf("*** main thread end\n");
}

// random generate route table
int rand_init_route_table(int **rt, int dim_x, int dim_y, int connect_num, int max_send_num);
void print_route_table(int **rt, int dim_x, int dim_y);
void init_topo_mesh(struct Router *rl, int dim_x, int dim_y);
void init_packets(struct Router *rl, int **rt, int dim_x, int dim_y, int total_packets_num);

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

    int total_packets_num = 0;
    int **route_table = (int **)malloc(sizeof(int *) * dim_x * dim_y);
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        route_table[i] = (int *)calloc(dim_x * dim_y, sizeof(int));
    }

    total_packets_num = rand_init_route_table(route_table, dim_x, dim_y, p_send, max_send_num);
    // route_table[0][1] = 10;
    // route_table[0][3] = 10;
    // route_table[1][0] = 10;
    // total_packets_num = 3;

    // print_route_table(route_table, dim_x, dim_y);

    channel_out = (struct ChannelOut *)calloc(dim_x * dim_y * CHANNEL_NUM, sizeof(struct ChannelOut));
    all_packets = (struct Packet *)malloc(total_packets_num * sizeof(struct Packet));
    struct Router *router_list = (struct Router *)calloc(dim_x * dim_y, sizeof(struct Router));

    init_topo_mesh(router_list, dim_x, dim_y);

    // init packets
    init_packets(router_list, route_table, dim_x, dim_y, total_packets_num);

    int cycle = 0;
    int *router_empty = (int *)calloc(THREAD_NUM, sizeof(int));
    int *router_busy = (int *)calloc(THREAD_NUM, sizeof(int));
    int *sync = (int *)calloc(THREAD_NUM * 2, sizeof(int));

    pthread_t thread_pid[THREAD_NUM];
    struct ThreadParam tp[THREAD_NUM];

    int any_busy = 0;

    printf("Thread num is : %d\n", THREAD_NUM);

    // start sim
    int inner_size = dim_x * dim_y / THREAD_NUM;
    if (dim_x * dim_y % THREAD_NUM != 0)
    {
        printf("Error: dim_x * dim_y %% THREAD_NUM != 0\n");
        exit(1);
    }
    for (int i = 0; i < THREAD_NUM; i++)
    {
        tp[i].pid = i;
        tp[i].size = inner_size;
        tp[i].router_list = router_list;
        tp[i].channel_out = channel_out;
        tp[i].router_empty = router_empty;
        tp[i].router_busy = router_busy;
        tp[i].sync = sync;
        tp[i].dim_x = dim_x;
        tp[i].dim_y = dim_y;
        // pthread_mutex_init(&p_lock1[i], NULL);
        // pthread_mutex_init(&p_lock2[i], NULL);
    }
    printf("Start ...\n");
    gettimeofday(&time_start, NULL);

    // creat thread
    int err;
    err = pthread_create(&thread_pid[0], NULL, (void *)main_thread, &tp[0]);
    if (err != 0)
        printf("pthread %d create error !\n", 0);
    for (int i = 1; i < THREAD_NUM; i++)
    {
        err = pthread_create(&thread_pid[i], NULL, (void *)one_thread, &tp[i]);
        if (err != 0)
            printf("pthread %d create error !\n", i);
    }

    // for (int i = 0; i < THREAD_NUM; i++)
    // {
    //     err = pthread_join(thread_pid[i], NULL);
    // }
    pthread_join(thread_pid[0], NULL);

    gettimeofday(&time_end, NULL);
    double total_time = ((double)(time_end.tv_sec) + (double)(time_end.tv_usec) / 1000000.0) - ((double)(time_start.tv_sec) + (double)(time_start.tv_usec) / 1000000.0);
    printf("Using time = %f s\n", total_time);

    return 0;
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
