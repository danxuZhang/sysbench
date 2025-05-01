#!/usr/bin/bash

get_cpu_name() {
    lscpu | grep "Model name" | cut -d':' -f2 | xargs
}
   
get_socket_count() {
    lscpu | grep "Socket(s)" | cut -d':' -f2 | xargs
}

get_core_count() {
    lscpu | grep "Core(s) per socket" | cut -d':' -f2 | xargs
}

get_total_mem_h() {
    free -h | grep Mem | awk '{print $2}'
}

get_free_mem_h() {
    free -h | grep Mem | awk '{print $4}'
}

get_hostname() {
    hostname
}

intel_mpi_run_on_socket() {
    local socket_id=$1
    local binary=$2
    shift 2  # Remove first two arguments to get the remaining arguments
    
    # Get cores per socket
    local cores_per_socket=$(get_core_count)
    
    # Calculate processor list for the specified socket
    local start_core=$((socket_id * cores_per_socket))
    local end_core=$(((socket_id + 1) * cores_per_socket - 1))
    local processor_list="${start_core}-${end_core}"
    
    echo "Running ${binary} on socket ${socket_id} (CPU cores ${processor_list})"
    echo "Arguments: $@"
    
    mpiexec -np ${cores_per_socket} \
            -genv I_MPI_PIN=1 \
            -genv I_MPI_PIN_DOMAIN=socket \
            -genv I_MPI_PIN_PROCESSOR_LIST=${processor_list} \
            ${binary} "$@"
    
    echo "Completed execution on socket ${socket_id}"
}

openmpi_run_on_socket() {
    local socket_id=$1
    local binary=$2
    shift 2  # Remove first two arguments to get the remaining arguments
    
    # Get cores per socket
    local cores_per_socket=$(get_core_count)
    local hostname=$(get_hostname)
    
    rankfile=gemm_rankfile
    rm -f ${rankfile}

    for ((i=0; i<$cores_per_socket; i++)); do     
        echo "rank $i=$hostname slot=$socket_id:$i" >> ${rankfile};
    done
    
    
    echo "Running ${binary} on socket ${socket_id} (CPU cores ${processor_list})"
    echo "Arguments: $@"
    
    mpirun -np $cores_per_socket \
	   --map-by rankfile:file=${rankfile} \
           --bind-to core \
           --report-bindings \
           ${binary} "$@"
   	
   rm ${rankfile} 
    echo "Completed execution on socket ${socket_id}"
}

# Example usage:
# openmpi_run_on_socket 0 ./bin/gemm_aocc_aocl_mpi 10 10 5
