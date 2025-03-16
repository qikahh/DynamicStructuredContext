DATA_START=82
DATA_ID=0
INFO=cross128

nohup python test_hierarchical_model.py \
    --gen_num 3 \
    --context Structure \
    --data_id ${DATA_ID} \
    --data_start ${DATA_START} \
    > output_st_3B_${INFO}_${DATA_START}.log 2>&1 &