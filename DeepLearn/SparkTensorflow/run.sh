#!/usr/bin/sh
hadoop fs -rmr hdfs://namenode.safe.lycc.qihoo.net:9000/tmp/sparkflow/dnn_demo/saveModel
/usr/bin/hadoop/software/spark-dl/bin/spark-submit \
    --master yarn-cluster \
    --executor-memory 6g \
    sparkFlowTrain.py \
    --data_path hdfs://namenode.safe.lycc.qihoo.net:9000/tmp/sparkflow/dnn_demo/data \
    --test_data_path hdfs://namenode.safe.lycc.qihoo.net:9000/tmp/sparkflow/dnn_demo/testdata \
    --save_path hdfs://namenode.safe.lycc.qihoo.net:9000/tmp/sparkflow/dnn_demo/saveModel \
    --worker_number 2 \
    --ps_number 1
