N_TRIALS=10
THRESHOLD=0.95

# bash argparse
while (( "$#" )); do
  case "$1" in
    --n-trials)
      N_TRIALS=$2
      shift 2
      ;;
    --threshold)
      THRESHOLD=$2
      shift 2
      ;;
    *) # preserve positional arguments
      shift
      ;;
  esac
done

docker run -it --rm --shm-size 8G --runtime=nvidia \
            -v $(pwd):/workspace \
            -v ${BASELOGDIR}:/baselogdir  \
            -v ${DATAPATH_RAW}:/data_raw \
            -v ${DATAPATH_CLEAN}:/data_clean \
            --env NVIDIA_VISIBLE_DEVICES=${GPUS} \
            -e RUN_CONFIG \
            catalyst-classification bash ./bin/run_autolabel.sh \
                    --data-raw /data_raw \
                    --data-clean /data_clean \
                    --baselogdir /baselogdir \
                    --n-trials ${N_TRIALS} \
                    --threshold ${THRESHOLD}
