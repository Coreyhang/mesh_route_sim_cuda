#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define DEPTH 5
#define DOUBLE_DEPTH 10
#define CHANNEL_NUM 4

// #define DEBUG

#ifdef DEBUG
#define debug(format, ...) printf(format, ##__VA_ARGS__)
#else
#define debug(format, ...)
#endif

// one route packet
struct Packet
{
    int dstx;
    int dsty;
};

struct Queue
{
    struct Packet **packets;
    int start;
    int end;
    int size;
};

struct Channel
{
    struct Queue *queue;
    struct Packet *out;
    int credit;
    int out_en;

    struct Channel *channel_connected;
};

struct Packet *channel_front(struct Channel *c)
{
    return c->queue->packets[c->queue->start % DEPTH];
}

void channel_pop(struct Channel *c)
{
    c->queue->start = (c->queue->start + 1) % DOUBLE_DEPTH;
    c->queue->size -= 1;
}

int channel_set_out(struct Channel *c, struct Packet *p)
{
    if ((c->out_en == 1) || (c->credit == 0))
    {
        return 0;
    }
    else
    {
        c->out = p;
        c->out_en = 1;
        c->credit -= 1;
        return 1;
    }
}

int channel_get_packet_from_channel_connected(struct Channel *c)
{
    if (c->channel_connected != NULL)
    {
        if (c->channel_connected->out_en == 1)
        {
            c->queue->packets[c->queue->end % DEPTH] = c->channel_connected->out;
            c->queue->end = (c->queue->end + 1) % DOUBLE_DEPTH;
            c->queue->size += 1;
            c->channel_connected->out_en = 0;
            c->channel_connected->credit = DEPTH - c->queue->size;
            return 1;
        }
        else
            c->channel_connected->credit = DEPTH - c->queue->size;
    }
    return 0;
}

struct Router
{
    int x;
    int y;
    struct Channel **channels; // left right up down
    struct Channel *local;
    int active;
    int empty;
};

void router_arbit(struct Router *r)
{
    r->active = 0;
    r->empty = 1;
    struct Packet *p = NULL;
    for (int i = 0; i < CHANNEL_NUM; i++)
    {
        if (r->channels[i]->queue->size > 0)
        { // left channel's queue not empty
            r->empty = 0;
            p = channel_front(r->channels[i]);
            if (p->dstx > r->x)
            {
                if (channel_set_out(r->channels[1], p))
                {
                    channel_pop(r->channels[i]);
                    r->active = 1;
                    debug("\t\t channel %d -> right\n", i);
                }
            }
            else if (p->dstx < r->x)
            {
                if (channel_set_out(r->channels[0], p))
                {
                    channel_pop(r->channels[i]);
                    r->active = 1;
                    debug("\t\t channel %d -> left\n", i);
                }
            }
            else if (p->dsty > r->y)
            {
                if (channel_set_out(r->channels[2], p))
                {
                    channel_pop(r->channels[i]);
                    r->active = 1;
                    debug("\t\t channel %d -> up\n", i);
                }
            }
            else if (p->dsty < r->y)
            {
                if (channel_set_out(r->channels[3], p))
                {
                    channel_pop(r->channels[i]);
                    r->active = 1;
                    debug("\t\t channel %d -> down\n", i);
                }
            }
            else if (r->local->out_en == 0)
            {
                r->local->out_en = 1;
                channel_pop(r->channels[i]);
                r->active = 1;
                debug("\t\t channel %d -> local\n", i);
            }
        }
    }
    if (r->local->queue->size > 0)
    { // local channel's queue not empty
        r->empty = 0;
        p = r->local->queue->packets[r->local->queue->start];
        if (p->dstx > r->x)
        {
            if (channel_set_out(r->channels[1], p))
            {
                r->local->queue->start = r->local->queue->start + 1;
                r->local->queue->size -= 1;
                r->active = 1;
                debug("\t\t local -> right\n");
            }
        }
        else if (p->dstx < r->x)
        {
            if (channel_set_out(r->channels[0], p))
            {
                r->local->queue->start = r->local->queue->start + 1;
                r->local->queue->size -= 1;
                r->active = 1;
                debug("\t\t local -> left\n");
            }
        }
        else if (p->dsty > r->y)
        {
            if (channel_set_out(r->channels[2], p))
            {
                r->local->queue->start = r->local->queue->start + 1;
                r->local->queue->size -= 1;
                r->active = 1;
                debug("\t\t local -> up\n");
            }
        }
        else if (p->dsty < r->y)
        {
            if (channel_set_out(r->channels[3], p))
            {
                r->local->queue->start = r->local->queue->start + 1;
                r->local->queue->size -= 1;
                r->active = 1;
                debug("\t\t local -> down\n");
            }
        }
        else if (r->local->out_en == 0)
        {
            r->local->out_en = 1;
            r->local->queue->start = r->local->queue->start + 1;
            r->local->queue->size -= 1;
            r->active = 1;
            debug("\t\t local -> local\n");
        }
    }
}

void router_trans(struct Router *r)
{
    for (int i = 0; i < CHANNEL_NUM; i++)
    {
        if (channel_get_packet_from_channel_connected(r->channels[i]))
        {
            debug("\t\t channel %d get\n", i);
        }
    }
    r->local->out_en = 0;
}

int main()
{
    time_t t;
    // srand((unsigned)time(&t));
    srand(1);

    struct timeval time_start, time_end;

    int dim_x = 10;
    int dim_y = 10;

    int p_send = 500; // 1/1000
    int max_send_num = 500;

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
                // route_table[i][j] = max_send_num;
            }
        }
    }

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

    struct Router **router_list = (struct Router **)malloc(sizeof(struct Router *) * dim_x * dim_y);

    for (int i = 0; i < dim_x * dim_y; i++)
    {
        router_list[i] = (struct Router *)malloc(sizeof(struct Router));
        router_list[i]->x = i % dim_x;
        router_list[i]->y = i / dim_x;
        router_list[i]->channels = (struct Channel **)malloc(sizeof(struct Channel *) * CHANNEL_NUM);
        for (int j = 0; j < CHANNEL_NUM; j++)
        {
            router_list[i]->channels[j] = (struct Channel *)malloc(sizeof(struct Channel));
            router_list[i]->channels[j]->credit = DEPTH;
            router_list[i]->channels[j]->out_en = 0;
            router_list[i]->channels[j]->out = NULL;
            router_list[i]->channels[j]->queue = (struct Queue *)malloc(sizeof(struct Queue));
            router_list[i]->channels[j]->queue->start = 0;
            router_list[i]->channels[j]->queue->end = 0;
            router_list[i]->channels[j]->queue->size = 0;
            router_list[i]->channels[j]->queue->packets = (struct Packet **)malloc(sizeof(struct Packet *) * DEPTH);
            router_list[i]->channels[j]->channel_connected = NULL;
        }
        router_list[i]->local = (struct Channel *)malloc(sizeof(struct Channel));
        router_list[i]->local->credit = DEPTH;
        router_list[i]->local->out_en = 0;
        router_list[i]->local->out = NULL;
        router_list[i]->local->queue = (struct Queue *)malloc(sizeof(struct Queue));
        router_list[i]->local->queue->start = 0;
        router_list[i]->local->queue->end = 0;
        router_list[i]->local->queue->size = 0;
        router_list[i]->local->queue->packets = NULL;
        router_list[i]->local->channel_connected = NULL;
    }

    for (int i = 0; i < dim_x * dim_y; i++)
    {
        int x = i % dim_x;
        int y = i / dim_x;
        // left right up down
        router_list[i]->channels[0]->channel_connected = (x == 0) ? NULL : router_list[(x - 1) + y * dim_x]->channels[1];
        router_list[i]->channels[1]->channel_connected = (x == dim_x - 1) ? NULL : router_list[(x + 1) + y * dim_x]->channels[0];
        router_list[i]->channels[2]->channel_connected = (y == dim_y - 1) ? NULL : router_list[x + (y + 1) * dim_x]->channels[3];
        router_list[i]->channels[3]->channel_connected = (y == 0) ? NULL : router_list[x + (y - 1) * dim_x]->channels[2];
        int send_num = 0;
        for (int j = 0; j < dim_x * dim_y; j++)
        {
            send_num += route_table[i][j];
        }
        router_list[i]->local->queue->end = send_num;
        router_list[i]->local->queue->size = send_num;
        router_list[i]->local->queue->packets = (struct Packet **)malloc(send_num * sizeof(struct Packet *));
        send_num = 0;
        for (int j = 0; j < dim_x * dim_y; j++)
        {
            for (int m = 0; m < route_table[i][j]; m++)
            {
                struct Packet *p = (struct Packet *)malloc(sizeof(struct Packet));
                p->dstx = j % dim_x;
                p->dsty = j / dim_x;
                router_list[i]->local->queue->packets[send_num] = p;
                send_num += 1;
            }
        }
    }

    // print
    // for (int i=0; i<dim_x*dim_y; i++){
    //     int x = i % dim_x;
    //     int y = i / dim_x;
    //     for (int j=0; j<router_list[i]->local->queue->size; j++){
    //         printf("(%d, %d) --> (%d, %d)\n", x, y, router_list[i]->local->queue->packets[j]->dstx, router_list[i]->local->queue->packets[j]->dsty);
    //     }
    // }

    gettimeofday(&time_start, NULL);
    // start sim
    int cycle = 0;
    int active = 0;
    int empty = 1;
    int dead_lock_cnt = 0;
    printf("Start ...\n");
    while (1)
    {
        for (int i = 0; i < dim_x * dim_y; i++)
        {
            debug("\t arbit (%d, %d)\n", i % dim_x, i / dim_x);
            router_arbit(router_list[i]);
            active = active | router_list[i]->active;
            empty = empty & router_list[i]->empty;
        }
        for (int i = 0; i < dim_x * dim_y; i++)
        {
            debug("\t trans (%d, %d)\n", i % dim_x, i / dim_x);
            router_trans(router_list[i]);
        }
        printf("-- cycle : %d\n", cycle);
        if (active == 0)
        {
            if (empty == 1)
            {
                printf("-- cycle : %d\n", cycle);
                printf("End ...\n");
                break;
            }
            else if (dead_lock_cnt > 10)
            {
                printf("-- cycle : %d\n", cycle);
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
        active = 0;
        empty = 1;
    }
    gettimeofday(&time_end, NULL);
    double total_time = ((double)(time_end.tv_sec) + (double)(time_end.tv_usec) / 1000000.0) - ((double)(time_start.tv_sec) + (double)(time_start.tv_usec) / 1000000.0);
    printf("Using time = %f s\n", total_time);

    return 0;
}