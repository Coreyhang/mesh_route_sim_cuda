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

// struct Queue
// {
//     struct Packet **packets;
//     int start;
//     int end;
//     int size;
// };

// struct Channel
// {
//     struct Queue *queue;
//     struct Packet *out;
//     int credit;
//     int out_en;

//     struct Channel *channel_connected;
// };

// struct Packet *channel_front(struct Channel *c)
// {
//     return c->queue->packets[c->queue->start % DEPTH];
// }

// void channel_pop(struct Channel *c)
// {
//     c->queue->start = (c->queue->start + 1) % DOUBLE_DEPTH;
//     c->queue->size -= 1;
// }

// int channel_set_out(struct Channel *c, struct Packet *p)
// {
//     if ((c->out_en == 1) || (c->credit == 0))
//     {
//         return 0;
//     }
//     else
//     {
//         c->out = p;
//         c->out_en = 1;
//         c->credit -= 1;
//         return 1;
//     }
// }

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

void router_arbit(struct Router *r)
{
    struct Packet *p = NULL;
    for (int i = 0; i < CHANNEL_NUM; i++)
    {
        if (r->channels_size[i] > 0)
        { // left channel's queue not empty
            router_empty = 0;
            p = r->channels[i][r->channels_start[i]];
            if (p->dstx > r->x)
            {
                if (channel_out[(r->id) * CHANNEL_NUM + 1].out == NULL && r->channels_credit[1] > 0)
                {
                    channel_out[(r->id) * CHANNEL_NUM + 1].out = p; // assign to ChannelOut
                    r->channels_credit[1] -= 1;
                    r->channels_start[i] = (r->channels_start[i] + 1) % DEPTH; // pop first packet
                    r->channels_size[i] -= 1;
                    channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                    router_busy = 1;
                    debug("\t\t channel %d -> right\n", i);
                }
                // else
                //     channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
            }
            else if (p->dstx < r->x)
            {
                if (channel_out[(r->id) * CHANNEL_NUM + 0].out == NULL && r->channels_credit[0] > 0)
                {
                    channel_out[(r->id) * CHANNEL_NUM + 0].out = p; // assign to ChannelOut
                    r->channels_credit[0] -= 1;
                    r->channels_start[i] = (r->channels_start[i] + 1) % DEPTH; // pop first packet
                    r->channels_size[i] -= 1;
                    channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                    router_busy = 1;
                    debug("\t\t channel %d -> left\n", i);
                }
                // else
                //     channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
            }
            else if (p->dsty > r->y)
            {
                if (channel_out[(r->id) * CHANNEL_NUM + 2].out == NULL && r->channels_credit[2] > 0)
                {
                    channel_out[(r->id) * CHANNEL_NUM + 2].out = p; // assign to ChannelOut
                    r->channels_credit[2] -= 1;
                    r->channels_start[i] = (r->channels_start[i] + 1) % DEPTH; // pop first packet
                    r->channels_size[i] -= 1;
                    channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                    router_busy = 1;
                    debug("\t\t channel %d -> up\n", i);
                }
                // else
                //     channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
            }
            else if (p->dsty < r->y)
            {
                if (channel_out[(r->id) * CHANNEL_NUM + 3].out == NULL && r->channels_credit[3] > 0)
                {
                    channel_out[(r->id) * CHANNEL_NUM + 3].out = p; // assign to ChannelOut
                    r->channels_credit[3] -= 1;
                    r->channels_start[i] = (r->channels_start[i] + 1) % DEPTH; // pop first packet
                    r->channels_size[i] -= 1;
                    channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                    router_busy = 1;
                    debug("\t\t channel %d -> down\n", i);
                }
                // else
                //     channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
            }
            else if (r->local_out == NULL)
            {
                r->local_out = p;
                r->channels_start[i] = (r->channels_start[i] + 1) % DEPTH; // pop first packet
                r->channels_size[i] -= 1;
                channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 1; // update
                router_busy = 1;
                debug("\t\t channel %d -> local\n", i);
            }
            else
            {
                channel_out[(r->id) * CHANNEL_NUM + i].delta_credit = 0; // update
            }
        }
    }
    if (r->local_size > 0)
    { // local channel's queue not empty
        router_empty = 0;
        p = &r->local[r->local_start];
        if (p->dstx > r->x)
        {
            if (channel_out[(r->id) * CHANNEL_NUM + 1].out == NULL && r->channels_credit[1] > 0)
            {
                channel_out[(r->id) * CHANNEL_NUM + 1].out = p; // assign to ChannelOut
                r->channels_credit[1] -= 1;
                if (p->num == 1)
                {
                    r->local_start += 1; // pop first packet
                    r->local_size -= 1;
                }
                else
                    p->num -= 1;
                router_busy = 1;
                debug("\t\t local -> right\n");
            }
        }
        else if (p->dstx < r->x)
        {
            if (channel_out[(r->id) * CHANNEL_NUM + 0].out == NULL && r->channels_credit[0] > 0)
            {
                channel_out[(r->id) * CHANNEL_NUM + 0].out = p; // assign to ChannelOut
                r->channels_credit[0] -= 1;
                if (p->num == 1)
                {
                    r->local_start += 1; // pop first packet
                    r->local_size -= 1;
                }
                else
                    p->num -= 1;
                router_busy = 1;
                debug("\t\t local -> left\n");
            }
        }
        else if (p->dsty > r->y)
        {
            if (channel_out[(r->id) * CHANNEL_NUM + 2].out == NULL && r->channels_credit[2] > 0)
            {
                channel_out[(r->id) * CHANNEL_NUM + 2].out = p; // assign to ChannelOut
                r->channels_credit[2] -= 1;
                if (p->num == 1)
                {
                    r->local_start += 1; // pop first packet
                    r->local_size -= 1;
                }
                else
                    p->num -= 1;
                router_busy = 1;
                debug("\t\t local -> up\n");
            }
        }
        else if (p->dsty < r->y)
        {
            if (channel_out[(r->id) * CHANNEL_NUM + 3].out == NULL && r->channels_credit[3] > 0)
            {
                channel_out[(r->id) * CHANNEL_NUM + 3].out = p; // assign to ChannelOut
                r->channels_credit[3] -= 1;
                if (p->num == 1)
                {
                    r->local_start += 1; // pop first packet
                    r->local_size -= 1;
                }
                else
                    p->num -= 1;
                router_busy = 1;
                debug("\t\t local -> down\n");
            }
        }
        else if (r->local_out == NULL)
        {
            r->local_out = p;
            if (p->num == 1)
            {
                r->local_start += 1; // pop first packet
                r->local_size -= 1;
            }
            else
                p->num -= 1;
            router_busy = 1;
            debug("\t\t local -> local\n");
        }
    }
}

inline void router_trans(struct Router *r)
{
    for (int i = 0; i < CHANNEL_NUM; i++)
    {
        if (r->channels_connected[i] != NULL)
        {
            r->channels_credit[i] += r->channels_connected[i]->delta_credit; // update credit
            r->channels_connected[i]->delta_credit = 0;
            if (r->channels_connected[i]->out != NULL)
            {
                r->channels[i][r->channels_end[i]] = r->channels_connected[i]->out; // push packet in input buffer
                r->channels_end[i] = (r->channels_end[i] + 1) % DEPTH;
                r->channels_size[i] += 1;
                r->channels_connected[i]->out = NULL;
                debug("\t\t channel %d get\n", i); // clear connected channel's out sign, prepare for next arbit
            }
        }
    }
    r->local_out = NULL; // clear local output buffer's out sign
}

int main()
{
    time_t t;
    // srand((unsigned)time(&t));
    srand(1);

    struct timeval time_start, time_end;

    int dim_x = 10;
    int dim_y = 10;

    int p_send = 1000; // 1/1000
    int max_send_num = 5;

    int total_packets_num = 0;
    int **route_table = (int **)malloc(sizeof(int *) * dim_x * dim_y);
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        route_table[i] = (int *)calloc(dim_x * dim_y, sizeof(int));
    }

    // random generate route table
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        for (int j = 0; j < dim_x * dim_y; j++)
        {
            if ((rand() % 1000) < p_send)
            {
                route_table[i][j] = rand() % max_send_num + 1;
                total_packets_num += 1;
            }
        }
    }

    // route_table[0][1] = 10;
    // total_packets_num = 1;

    // print
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
            printf("%d\t", route_table[i][j]);
        }
        printf("\n");
    }

    channel_out = (struct ChannelOut *)calloc(dim_x * dim_y * CHANNEL_NUM, sizeof(struct ChannelOut));
    all_packets = (struct Packet *)malloc(total_packets_num * sizeof(struct Packet));
    struct Router *router_list = (struct Router *)calloc(dim_x * dim_y, sizeof(struct Router));

    for (int i = 0; i < dim_x * dim_y; i++)
    {
        router_list[i].x = i % dim_x;
        router_list[i].y = i / dim_x;
        router_list[i].id = i;
        for (int j = 0; j < CHANNEL_NUM; j++)
            router_list[i].channels_credit[j] = DEPTH;
    }

    // init connection
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        int x = i % dim_x;
        int y = i / dim_x;
        // left right up down
        router_list[i].channels_connected[0] = (x == 0) ? NULL : &channel_out[(x - 1 + y * dim_x) * CHANNEL_NUM + 1];
        router_list[i].channels_connected[1] = (x == dim_x - 1) ? NULL : &channel_out[(x + 1 + y * dim_x) * CHANNEL_NUM + 0];
        router_list[i].channels_connected[2] = (y == dim_y - 1) ? NULL : &channel_out[(x + (y + 1) * dim_x) * CHANNEL_NUM + 3];
        router_list[i].channels_connected[3] = (y == 0) ? NULL : &channel_out[(x + (y - 1) * dim_x) * CHANNEL_NUM + 2];
    }
    // init packets
    int packet_ptr = 0;
    for (int i = 0; i < dim_x * dim_y; i++)
    {
        router_list[i].local = &all_packets[packet_ptr];
        router_list[i].local_size = 0;
        // local
        for (int j = 0; j < dim_x * dim_y; j++)
        {
            if (route_table[i][j] > 0)
            {
                all_packets[packet_ptr].dstx = j % dim_x;
                all_packets[packet_ptr].dsty = j / dim_x;
                all_packets[packet_ptr].num = route_table[i][j];
                router_list[i].local_size += 1;
                packet_ptr += 1;
            }
        }
    }
    assert(packet_ptr == total_packets_num);

    // print
    // for (int i = 0; i < dim_x * dim_y; i++)
    // {
    //     int x = i % dim_x;
    //     int y = i / dim_x;
    //     for (int j = 0; j < router_list[i].local_size; j++)
    //     {
    //         printf("(%d, %d) --> (%d, %d) x %d\n", x, y, router_list[i].local[j].dstx, router_list[i].local[j].dsty, router_list[i].local[j].num);
    //         printf("%d\n", router_list[i].local_size);
    //         assert(router_list[i].local[j].num > 0);
    //     }
    // }

    gettimeofday(&time_start, NULL);
    // start sim
    int cycle = 0;
    int dead_lock_cnt = 0;
    printf("Start ...\n");
    while (1)
    {
        for (int i = 0; i < dim_x * dim_y; i++)
        {
            debug("\t arbit (%d, %d)\n", i % dim_x, i / dim_x);
            router_arbit(&router_list[i]);
        }
        for (int i = 0; i < dim_x * dim_y; i++)
        {
            debug("\t trans (%d, %d)\n", i % dim_x, i / dim_x);
            router_trans(&router_list[i]);
        }
        printf("-- cycle : %d\n", cycle);
        if (router_busy == 0)
        {
            if (router_empty == 1)
            {
                printf("Total cycle : %d\n", cycle);
                printf("End ...\n");
                break;
            }
            else if (dead_lock_cnt > 10)
            {
                printf("Total cycle : %d\n", cycle);
                printf("Dead Lock ...\n");
                break;
            }
            else
            {
                dead_lock_cnt += 1;
            }
        }
        else
        {
            dead_lock_cnt = 0;
        }
        cycle += 1;
        router_busy = 0;
        router_empty = 1;
    }
    gettimeofday(&time_end, NULL);
    double total_time = ((double)(time_end.tv_sec) + (double)(time_end.tv_usec) / 1000000.0) - ((double)(time_start.tv_sec) + (double)(time_start.tv_usec) / 1000000.0);
    printf("Using time = %f s\n", total_time);

    return 0;
}