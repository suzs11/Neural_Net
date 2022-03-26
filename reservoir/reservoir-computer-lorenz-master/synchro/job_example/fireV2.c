/* fire_mpi.c - final step
 * set up probabilities for each process, add MPI commands
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

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

    if (argc==2) {
        n_trials = atoi(argv[1]);
    }

    // setup MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    mypid = getpid();

    char* PBS_JOBID = NULL;
    if ((PBS_JOBID = getenv("PBS_JOBID")) != NULL) {

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

    } else if ((PBS_JOBID = getenv("SLURM_JOB_ID")) != NULL) {

    	char* SLURM_JOB_NAME = getenv("SLURM_JOB_NAME");
    	char* SLURM_JOB_ACCOUNT = getenv("SLURM_JOB_ACCOUNT");
    	char* SLURM_JOB_PARTITION = getenv("SLURM_JOB_PARTITION");
    	char* SLURM_CLUSTER_NAME = getenv("SLURM_CLUSTER_NAME");
    	char* SLURM_JOB_NODELIST = getenv("SLURM_JOB_NODELIST");
    	char* SLURM_JOB_NUM_NODES = getenv("SLURM_JOB_NUM_NODES");
    	char* SLURM_LOCALID = getenv("SLURM_LOCALID");
    	char* SLURM_NODEID = getenv("SLURM_NODEID");
    	char* SLURM_NTASKS = getenv("SLURM_NTASKS");
    	char* SLURMD_NODENAME = getenv("SLURMD_NODENAME");
    	char* SLURM_CPUS_ON_NODE = getenv("SLURM_CPUS_ON_NODE");
    	char* SLURM_CPUS_PER_TASK = getenv("SLURM_CPUS_PER_TASK");
    	char* SLURM_GTIDS = getenv("SLURM_GTIDS");
    	char* SLURM_JOB_CPUS_PER_NODE = getenv("SLURM_JOB_CPUS_PER_NODE");
    	char* SLURM_JOB_QOS = getenv("SLURM_JOB_QOS");
    	char* HIPS_VISIBLE_DEVICES = getenv("HIPS_VISIBLE_DEVICES");
    	char* GPU_DEVICE_ORDINAL = getenv("GPU_DEVICE_ORDINAL");
    	char* OFFLOAD_DEVICES = getenv("OFFLOAD_DEVICES");
    	char* SLURM_JOB_DCUS = getenv("SLURM_JOB_DCUS");
    	char* SLURM_STEP_DCUS = getenv("SLURM_STEP_DCUS");
    	char* SLURM_JOB_RESERVATION = getenv("SLURM_JOB_RESERVATION");
    	char* SLURM_MEM_PER_CPU = getenv("SLURM_MEM_PER_CPU");
    	char* SLURM_MEM_PER_NODE = getenv("SLURM_MEM_PER_NODE");
    	char* SLURM_NODE_ALIASES = getenv("SLURM_NODE_ALIASES");
    	char* SLURM_NTASKS_PER_CORE = getenv("SLURM_NTASKS_PER_CORE");
    	char* SLURM_NTASKS_PER_NODE = getenv("SLURM_NTASKS_PER_NODE");
    	char* SLURM_NTASKS_PER_SOCKET = getenv("SLURM_NTASKS_PER_SOCKET");
    	char* SLURM_PACK_SIZE = getenv("SLURM_PACK_SIZE");
    	char* SLURM_PRIO_PROCESS = getenv("SLURM_PRIO_PROCESS");
    	char* SLURM_PROCID = getenv("SLURM_PROCID");
    	char* SLURM_PROFILE = getenv("SLURM_PROFILE");
    	char* SLURM_RESTART_COUNT = getenv("SLURM_RESTART_COUNT");
    	char* SLURM_SUBMIT_DIR = getenv("SLURM_SUBMIT_DIR");
    	char* SLURM_SUBMIT_HOST = getenv("SLURM_SUBMIT_HOST");
    	char* SLURM_TASKS_PER_NODE = getenv("SLURM_TASKS_PER_NODE");
    	char* SLURM_TASK_PID = getenv("SLURM_TASK_PID");
    	char* SLURM_TOPOLOGY_ADDR = getenv("SLURM_TOPOLOGY_ADDR");
    	char* SLURM_TOPOLOGY_ADDR_PATTERN = getenv("SLURM_TOPOLOGY_ADDR_PATTERN");
    	char* SLURM_DISTRIBUTION = getenv("SLURM_DISTRIBUTION");

    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_ID = %s\n",rank, mypid, PBS_JOBID);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_NAME = %s\n",rank, mypid, SLURM_JOB_NAME);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_ACCOUNT = %s\n",rank, mypid, SLURM_JOB_ACCOUNT);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_PARTITION = %s\n",rank, mypid, SLURM_JOB_PARTITION);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_CLUSTER_NAME = %s\n",rank, mypid, SLURM_CLUSTER_NAME);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_NUM_NODES = %s\n",rank, mypid, SLURM_JOB_NUM_NODES);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_NODELIST = %s\n",rank, mypid, SLURM_JOB_NODELIST);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_LOCALID = %s\n",rank, mypid, SLURM_LOCALID);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_NODEID = %s\n",rank, mypid, SLURM_NODEID);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_NTASKS = %s\n",rank, mypid, SLURM_NTASKS);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURMD_NODENAME = %s\n",rank, mypid, SLURMD_NODENAME);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_CPUS_ON_NODE = %s\n",rank, mypid, SLURM_CPUS_ON_NODE);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_CPUS_PER_TASK = %s\n",rank, mypid, SLURM_CPUS_PER_TASK);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_GTIDS = %s\n",rank, mypid, SLURM_GTIDS);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_CPUS_PER_NODE = %s\n",rank, mypid, SLURM_JOB_CPUS_PER_NODE);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_QOS = %s\n",rank, mypid, SLURM_JOB_QOS);
    	printf("PID[%d] = %d, Montro Carlo Fire, HIPS_VISIBLE_DEVICES = %s\n",rank, mypid, HIPS_VISIBLE_DEVICES);
    	printf("PID[%d] = %d, Montro Carlo Fire, GPU_DEVICE_ORDINAL = %s\n",rank, mypid, GPU_DEVICE_ORDINAL);
    	printf("PID[%d] = %d, Montro Carlo Fire, OFFLOAD_DEVICES = %s\n",rank, mypid, OFFLOAD_DEVICES);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_DCUS = %s\n",rank, mypid, SLURM_JOB_DCUS);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_STEP_DCUS = %s\n",rank, mypid, SLURM_STEP_DCUS);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_JOB_RESERVATION = %s\n",rank, mypid, SLURM_JOB_RESERVATION);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_MEM_PER_CPU = %s\n",rank, mypid, SLURM_MEM_PER_CPU);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_MEM_PER_NODE = %s\n",rank, mypid, SLURM_MEM_PER_NODE);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_NODE_ALIASES = %s\n",rank, mypid, SLURM_NODE_ALIASES);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_NTASKS_PER_CORE = %s\n",rank, mypid, SLURM_NTASKS_PER_CORE);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_NTASKS_PER_NODE = %s\n",rank, mypid, SLURM_NTASKS_PER_NODE);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_NTASKS_PER_SOCKET = %s\n",rank, mypid, SLURM_NTASKS_PER_SOCKET);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_PACK_SIZE = %s\n",rank, mypid, SLURM_PACK_SIZE);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_PRIO_PROCESS = %s\n",rank, mypid, SLURM_PRIO_PROCESS);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_PROCID = %s\n",rank, mypid, SLURM_PROCID);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_PROFILE = %s\n",rank, mypid, SLURM_PROFILE);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_RESTART_COUNT = %s\n",rank, mypid, SLURM_RESTART_COUNT);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_SUBMIT_DIR = %s\n",rank, mypid, SLURM_SUBMIT_DIR);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_SUBMIT_HOST = %s\n",rank, mypid, SLURM_SUBMIT_HOST);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_TASKS_PER_NODE = %s\n",rank, mypid, SLURM_TASKS_PER_NODE);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_TASK_PID = %s\n",rank, mypid, SLURM_TASK_PID);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_TOPOLOGY_ADDR = %s\n",rank, mypid, SLURM_TOPOLOGY_ADDR);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_TOPOLOGY_ADDR_PATTERN = %s\n",rank, mypid, SLURM_TOPOLOGY_ADDR_PATTERN);
    	printf("PID[%d] = %d, Montro Carlo Fire, SLURM_DISTRIBUTION = %s\n",rank, mypid, SLURM_DISTRIBUTION);

    } else {

    	printf("PID[%d] = %d, Montro Carlo Fire, Unknown RMS type\n",rank, mypid);
    }

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

    // clean up
    delete_forest(forest_size,forest);
    free(per_burns);
    MPI_Finalize();


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
