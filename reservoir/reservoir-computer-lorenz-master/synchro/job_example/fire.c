/* fire_mpi.c - final step
 * set up probabilities for each process, add MPI commands
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

/*
#include <mpi.h>
*/
#include <mpi.h>

#define UNBURNT 0
#define SMOLDERING 1
#define BURNING 2
#define BURNT 3

#define true 1
#define false 0

typedef int boolean;

extern void seed_by_time(int);
extern int ** allocate_forest(int);
extern void initialize_forest(int, int **);
extern double get_percent_burned(int, int **);
extern void delete_forest(int, int **);
extern void light_tree(int, int **,int,int);
extern boolean forest_is_burning(int, int **);
extern void forest_burns(int, int **,double);
extern void burn_until_out(int,int **,double,int,int);
extern void print_forest(int, int **);

MPI_Status status;
int rank;
int size;

#define min(X,Y) ((X) < (Y) ? (X) : (Y))

int main(int argc, char ** argv) {
    // initial conditions and variable definitions
    int forest_size=20;
    double prob_spread;
    double prob_min=0.0;
    double prob_max=1.0;
    double prob_step;
    int **forest;
    // add second loop index
    int i,j;
    double percent_burned;
    int i_trial;
    int n_trials=600000;
    int i_prob;
    int i_start,i_finish;
    int n_probs=100;
    double * per_burns;
    // add storage array
    double * per_storage;
    int mypid = 0;

    struct timeval tvBegin, tvEnd;
    double dDuration = 0.0;

    if (argc==2) {
        n_trials = atoi(argv[1]);
    }

    gettimeofday(&tvBegin, NULL);

    // setup MPI
    MPI_Init(&argc, &argv);

    gettimeofday(&tvEnd, NULL);
    //<1>.获得毫秒数:
    //dDuration = 1000 * (tvEnd.tv_sec - tvBegin.tv_sec) + ((tvEnd.tv_usec - tvBegin.tv_usec) / 1000.0);
    //<2>.获得秒数:
    dDuration = (tvEnd.tv_sec - tvBegin.tv_sec) + ((tvEnd.tv_usec - tvBegin.tv_usec) / 1000.0) / 1000.0;

    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

/*
    char* PBS_JOBID = getenv("PBS_JOBID");
    char* PBS_JOBCOOKIE = getenv("PBS_JOBCOOKIE");
    char* PBS_NODENUM = getenv("PBS_NODENUM");
    char* PBS_TASKNUM = getenv("PBS_TASKNUM");
    char* PBS_MOMPORT = getenv("PBS_MOMPORT");
    char* PBS_NODEFILE = getenv("PBS_NODEFILE");
    char* PBS_GPUFILE = getenv("PBS_GPUFILE");
    char* PBS_VNODENUM = getenv("PBS_VNODENUM");
    char* PBS_NNODES = getenv("PBS_NNODES");
    char* PBS_NUM_NODES = getenv("PBS_NUM_NODES");
    char* PBS_NUM_PPN = getenv("PBS_NUM_PPN");
    char* PBS_NP = getenv("PBS_NP");
    char* PBS_EXECGPUS = getenv("PBS_EXECGPUS");
    char* CUDA_VISIBLE_DEVICES = getenv("CUDA_VISIBLE_DEVICES");
    char* GPU_DEVICE_ORDINAL = getenv("GPU_DEVICE_ORDINAL");
    char* OFFLOAD_DEVICES = getenv("OFFLOAD_DEVICES");
*/

/*
    char* = getenv("PBS_NODENUM");
    char* = getenv("PBS_NODENUM");
    char* = getenv("PBS_NODENUM");
*/

    mypid = getpid();
    printf("PID[%d] = %d, Montro Carlo Fire, time of MPI_Init = %.8lf\n",rank, mypid, dDuration);

/*
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_JOBID = %s\n",rank, mypid, PBS_JOBID);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_NODENUM = %s\n",rank, mypid, PBS_NODENUM);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_TASKNUM = %s\n",rank, mypid, PBS_TASKNUM);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_MOMPORT = %s\n",rank, mypid, PBS_MOMPORT);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_NODEFILE = %s\n",rank, mypid, PBS_NODEFILE);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_GPUFILE = %s\n",rank, mypid, PBS_GPUFILE);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_VNODENUM = %s\n",rank, mypid, PBS_VNODENUM);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_NNODES = %s\n",rank, mypid, PBS_NNODES);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_NUM_NODES = %s\n",rank, mypid, PBS_NUM_NODES);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_NUM_PPN = %s\n",rank, mypid, PBS_NUM_PPN);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_NP = %s\n",rank, mypid, PBS_NP);
    printf("PID[%d] = %d, Montro Carlo Fire, PBS_EXECGPUS = %s\n",rank, mypid, PBS_EXECGPUS);
    printf("PID[%d] = %d, Montro Carlo Fire, CUDA_VISIBLE_DEVICES = %s\n",rank, mypid, CUDA_VISIBLE_DEVICES);
    printf("PID[%d] = %d, Montro Carlo Fire, GPU_DEVICE_ORDINAL = %s\n",rank, mypid, GPU_DEVICE_ORDINAL);
    printf("PID[%d] = %d, Montro Carlo Fire, OFFLOAD_DEVICES = %s\n",rank, mypid, OFFLOAD_DEVICES);
*/

    // setup problem, allocate memory
    seed_by_time(0);
    forest=allocate_forest(forest_size);
    per_burns = (double *)malloc(sizeof(double)*n_probs+1);
    // allocate storage array
    per_storage = (double *)malloc(sizeof(double)*n_probs+1);

    // for a number of probabilities, calculate
    // average burn and output
    prob_step = (prob_max-prob_min)/(double)(n_probs-1);
    // change loop to go from 0 to max
    for (i_prob = 0 ; i_prob < n_probs; i_prob++) {
        //for a number of trials, calculate average
        //percent burn
        prob_spread = prob_min + (double)i_prob * prob_step;
        percent_burned=0.0;
        // change loop to do fewer trials
        for (i_trial=0; i_trial < n_trials/size; i_trial++) {
            //burn until fire is gone
            burn_until_out(forest_size,forest,prob_spread,10,10);
            percent_burned+=get_percent_burned(forest_size,forest);
        }
        percent_burned/=n_trials;
        per_burns[i_prob]=percent_burned;

    }

    // communicate
    if (rank>0) {
        // send values from client to server
        MPI_Send(per_burns,n_probs,MPI_DOUBLE,
            0,rank,MPI_COMM_WORLD);
    } else {
        // for every client, recieve values into a buffer
        // so that you do not overwrite old calculations,
        // and update calculated values
        for (i = 1; i< size; i++) {
            MPI_Recv(per_storage,n_probs,MPI_DOUBLE,
                i,i,MPI_COMM_WORLD,&status);
            for (j=0;j<n_probs;j++) {
                per_burns[j]+=per_storage[j];
            }
        }
        // average over clients
        //for (j=0;j<n_probs;j++) {
            //per_burns[j]/=size;
        //}
        // print output
        printf("Probability of fire spreading, Average percent burned\n");
        for (i_prob =0 ; i_prob<n_probs; i_prob++) {
            prob_spread = prob_min + (double)i_prob * prob_step;
            printf("%lf , %lf\n",prob_spread,per_burns[i_prob]);
        }
    }

    gettimeofday(&tvBegin, NULL);
    dDuration = (tvBegin.tv_sec - tvEnd.tv_sec) + ((tvBegin.tv_usec - tvEnd.tv_usec) / 1000.0) / 1000.0;

    printf("PID[%d] = %d, Montro Carlo Fire, time of MPI_SendRecv = %.8lf\n",rank, mypid, dDuration);

    // clean up
    delete_forest(forest_size,forest);
    free(per_burns);

    gettimeofday(&tvBegin, NULL);
    MPI_Finalize();
    gettimeofday(&tvEnd, NULL);
    dDuration = (tvEnd.tv_sec - tvBegin.tv_sec) + ((tvEnd.tv_usec - tvBegin.tv_usec) / 1000.0) / 1000.0;

    printf("PID[%d] = %d, Montro Carlo Fire, time of MPI_Finalize = %.8lf\n",rank, mypid, dDuration);
    exit(0);

}

void seed_by_time(int offset) {
    time_t the_time;
    time(&the_time);
    srand((int)the_time+offset);
}

void burn_until_out(int forest_size,int ** forest, double prob_spread,
    int start_i, int start_j) {

    initialize_forest(forest_size,forest);
    light_tree(forest_size,forest,start_i,start_j);

    // burn until fire is gone
    while(forest_is_burning(forest_size,forest)) {
        forest_burns(forest_size,forest,prob_spread);
    }
}

double get_percent_burned(int forest_size,int ** forest) {
    int i,j;
    int total = forest_size*forest_size-1;
    int sum=0;

    // calculate pecrent burned
    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i][j]==BURNT) {
                sum++;
            }
        }
    }

    // return percent burned;
    return ((double)(sum-1)/(double)total);
}


int ** allocate_forest(int forest_size) {
    int i,j;
    int ** forest;

    forest = (int **) malloc (sizeof(int*)*forest_size);
    for (i=0;i<forest_size;i++) {
        forest[i] = (int *) malloc (sizeof(int)*forest_size);
    }

    return forest;
}

void initialize_forest(int forest_size, int ** forest) {
    int i,j;

    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
            forest[i][j]=UNBURNT;
        }
    }
}

void delete_forest(int forest_size, int ** forest) {
    int i;

    for (i=0;i<forest_size;i++) {
        free(forest[i]);
    }
    free(forest);
}

void light_tree(int forest_size, int ** forest, int i, int j) {
    forest[i][j]=SMOLDERING;
}

boolean fire_spreads(double prob_spread) {
    if ((double)rand()/(double)RAND_MAX < prob_spread) 
        return true;
    else
        return false;
}

void forest_burns(int forest_size, int **forest,double prob_spread) {
    int i,j;
    extern boolean fire_spreads(double);

    //burning trees burn down, smoldering trees ignite
    for (i=0; i<forest_size; i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i][j]==BURNING) forest[i][j]=BURNT;
            if (forest[i][j]==SMOLDERING) forest[i][j]=BURNING;
        }
    }

    //unburnt trees catch fire
    for (i=0; i<forest_size; i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i][j]==BURNING) {
                if (i!=0) { // North
                    if (fire_spreads(prob_spread)&&forest[i-1][j]==UNBURNT) {
                        forest[i-1][j]=SMOLDERING;
                    }
                }
                if (i!=forest_size-1) { //South
                    if (fire_spreads(prob_spread)&&forest[i+1][j]==UNBURNT) {
                        forest[i+1][j]=SMOLDERING;
                    }
                }
                if (j!=0) { // West
                    if (fire_spreads(prob_spread)&&forest[i][j-1]==UNBURNT) {
                        forest[i][j-1]=SMOLDERING;
                    }
                }
                if (j!=forest_size-1) { // East
                    if (fire_spreads(prob_spread)&&forest[i][j+1]==UNBURNT) {
                        forest[i][j+1]=SMOLDERING;
                    }
                }
            }
        }
    }
}

boolean forest_is_burning(int forest_size, int ** forest) {
    int i,j;

    for (i=0; i<forest_size; i++) {
        for (j=0; j<forest_size; j++) {
            if (forest[i][j]==SMOLDERING||forest[i][j]==BURNING) {
                return true;
            }
        }
    }
    return false;
}

void print_forest(int forest_size,int ** forest) {
    int i,j;

    for (i=0;i<forest_size;i++) {
        for (j=0;j<forest_size;j++) {
            if (forest[i][j]==BURNT) {
                printf(".");
            } else {
                printf("X");
            }
        }
        printf("\n");
    }
}
