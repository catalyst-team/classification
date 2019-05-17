docker run -it  \
            -v $(pwd):/workspace \
	        -v ${OUT_DIR}:/logdir  \
	        -v ${IMAGES_DIR}:/data_clean \
            --env IMAGES_DIR=/data_clean \
            --env OUT_DIR=/logdir \
            --env NVIDIA_VISIBLE_DEVICES=${GPUS} \
            -e RUN_CONFIG \
            catalyst-classification bash ./bin/run_autoclassification.sh