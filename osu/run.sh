#!/usr/bin/env bash

BUILD_PREFIX="${BUILD_PREFIX:-build}"
EXEC_PREFIX="${EXEC_PREFIX:-libexec/osu-micro-benchmarks/mpi}"
OSU_BW="${OSU_BW:-${BUILD_PREFIX}/${EXEC_PREFIX}/pt2pt/osu_bw}"
OSU_LAT="${OSU_LAT:-${BUILD_PREFIX}/${EXEC_PREFIX}/pt2pt/osu_latency}"
OSU_ALLREDUCE="${OSU_ALLREDUCE:-${BUILD_PREFIX}/${EXEC_PREFIX}/collective/osu_allreduce}"
OSU_BCAST="${OSU_BCAST:-${BUILD_PREFIX}/${EXEC_PREFIX}/collective/osu_bcast}"
OSU_ALLTOALL="${OSU_ALLTOALL:-${BUILD_PREFIX}/${EXEC_PREFIX}/collective/osu_alltoall}"

OUTPUT_DIR="${OUTPUT_DIR:-./results_$(date +%Y%m%d_%H%M%S)}"
BW_CSV="${BW_CSV:-${OUTPUT_DIR}/bw.csv}"
LAT_CSV="${LAT_CSV:-${OUTPUT_DIR}/lat.csv}"
COLL_CSV="${COLL_CSV:-${OUTPUT_DIR}/coll.csv}"
COLL_LOCAL_CSV="${COLL_LOCAL_CSV:-${OUTPUT_DIR}/coll_local.csv}"

MPI_ARGS="${MPI_ARGS:- --map-by node}"
BW_ARGS="${BW_ARGS:---type mpi_char -m 1:4194304 -i 1000}"
LAT_ARGS="${LAT_ARGS:---type mpi_char -m 1:4194304 -i 1000 -z 99,95,50}"
COLL_ARGS="${COLL_ARGS:---type mpi_float -m 1:1048576 -i 100 -z 99,95,50}"

NODES=(
    "coffeepot0"
    "coffeepot1"
    "coffeepot2"
)
NPERNODE=192

setup() {
    module purge
    module load openmpi/5.0.3-gcc12.3-ucx1.17

    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
    fi

    echo "Running OSU Micro-Benchmarks for MPI"
    echo "OSU_BW: $OSU_BW"
    echo "OSU_LAT: $OSU_LAT"
    echo "OSU_ALLREDUCE: $OSU_ALLREDUCE"
    echo "OSU_BCAST: $OSU_BCAST"
    echo "OSU_ALLTOALL: $OSU_ALLTOALL"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
}

test_bandwidth_pair() {
    local src="$1"
    local dst="$2"

    mpiexec -host "$src,$dst" \
        ${MPI_ARGS} "${OSU_BW}" ${BW_ARGS} \
    | grep -v -e '^#' -e 'WARN' -e '^\[' -e '^$' \
    | awk -v src="$src" -v dst="$dst" \
        -v OFS=',' \
        '{print src, dst, $1, $2}' 
}

benchmark_bandwidth() {
    if [ ! -f "$BW_CSV" ]; then
        echo "Source,Destination,Size,Bandwidth(MB/s)" \
            | tee -a "$BW_CSV"
    fi
    for i in ${!NODES[@]}; do
        for j in ${!NODES[@]}; do
            if [ "$i" -ne "$j" ]; then
                echo "Bandwidth test ${NODES[$i]} -> ${NODES[$j]}"
                test_bandwidth_pair "${NODES[$i]}" "${NODES[$j]}" \
                    | tee -a "${BW_CSV}" 
                echo ""
            fi
        done
    done
}

test_latency_pair() {
    local src="$1"
    local dst="$2"

    mpiexec -host "$src,$dst" \
        ${MPI_ARGS} "${OSU_LAT}" ${LAT_ARGS} \
    | grep -v -e '^#' -e 'WARN' -e '^\[' -e '^$' \
    | awk -v src="$src" -v dst="$dst" \
        -v OFS=',' \
        '{print src, dst, $1, $2, $3, $4, $5}' 
}


benchmark_latency() {
    if [ ! -f "$LAT_CSV" ]; then
        echo "src,dst,Size,Avg Latency(us),P50 Tail Lat(us),P90 Tail Lat(us),P99 Tail Lat(us)" > "$LAT_CSV"
    fi
    for i in ${!NODES[@]}; do
        for j in ${!NODES[@]}; do
            if [ "$i" -ne "$j" ]; then
                echo "Latency test ${NODES[$i]} -> ${NODES[$j]}"
                test_latency_pair "${NODES[$i]}" "${NODES[$j]}" \
                    | tee -a "${LAT_CSV}" 
                echo ""
            fi
        done
    done
}

test_collective() {
    local coll_bin="$1"
    local nodes="$2"
    local npernode="$3"
    local hostfile="$4"
    local np=$((nodes * npernode))
    local func_name="$(basename ${coll_bin} | cut -d'_' -f2)"
    
    mpiexec -np "$np" \
        --hostfile "$hostfile" \
        ${MPI_ARGS} "${coll_bin}" ${COLL_ARGS} \
    | grep -v -e '^#' -e 'WARN' -e '^$' \
    | awk -v OFS=',' -v func_name="${func_name}" -v nodes="${nodes}" -v npernode="$npernode" \
        '{print func_name, nodes, npernode, $1, $2, $3, $4, $5, $6}'
}

test_collective_local() {
    local coll_bin="$1"
    local node="$2"
    local npernode="$3"
    local func_name="$(basename ${coll_bin} | cut -d'_' -f2)"

    mpiexec -np "$npernode" \
        --host "${node}:${npernode}" \
        ${MPI_ARGS} "${coll_bin}" ${COLL_ARGS} \
    | grep -v -e '^#' -e 'WARN' -e '^$' \
    | awk -v OFS=',' -v func_name="${func_name}" -v node="$node" -v npernode="$npernode" \
        '{print func_name, node, npernode, $1, $2, $3, $4, $5, $6}'
}

benchmark_collective() {
    echo "Benchmarking collectives"
    # Generate hostfile
    HOSTFILE="hostfile_tmp"
    touch "$HOSTFILE"
    echo "" > "$HOSTFILE"
    for node in "${NODES[@]}"; do
        echo "${node} slots=$NPERNODE" >> "$HOSTFILE"
    done


    if [ ! -f "$COLL_CSV" ]; then
        echo "Function,Nodes,NPerNode,Size,Avg Latency(us),P50 Tail Lat(us),P90 Tail Lat(us),P99 Tail Lat(us)" \
                | tee -a "$COLL_CSV"
    fi

    local n="${#NODES[@]}"

    test_collective "${OSU_ALLREDUCE}" $n $NPERNODE "$HOSTFILE" \
        | tee -a "$COLL_CSV"
    test_collective "${OSU_BCAST}" $n $NPERNODE "$HOSTFILE" \
        | tee -a "$COLL_CSV"
    test_collective "${OSU_ALLTOALL}" $n $NPERNODE "$HOSTFILE" \
        | tee -a "$COLL_CSV"

    rm "$HOSTFILE"
}

benchmark_collective_local() {
    echo "Benchmarking local collectives"
    if [ ! -f "$COLL_LOCAL_CSV" ]; then
        echo "Function,Node,NPerNode,Size,Avg Latency(us),P50 Tail Lat(us),P90 Tail Lat(us),P99 Tail Lat(us)" \
            | tee -a "$COLL_LOCAL_CSV"
    fi

    for node in "${NODES[@]}"; do
        test_collective_local "${OSU_ALLREDUCE}" "$node" $NPERNODE \
            | tee -a "$COLL_LOCAL_CSV"
        test_collective_local "${OSU_BCAST}" "$node" $NPERNODE \
            | tee -a "$COLL_LOCAL_CSV"
        test_collective_local "${OSU_ALLTOALL}" "$node" $NPERNODE \
            | tee -a "$COLL_LOCAL_CSV"
    done
}

setup
benchmark_bandwidth
benchmark_latency
benchmark_collective_local
benchmark_collective
