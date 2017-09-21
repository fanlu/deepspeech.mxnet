#!/bin/bash
# set -x
docker run --net=host --volume-driver=nvidia-docker --volume=nvidia_driver_375.20:/usr/local/nvidia:ro \
 --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools \
 --device=/dev/nvidia0 --device=/dev/nvidia1 --device=/dev/nvidia2 --device=/dev/nvidia3 \
 -e DMLC_PS_ROOT_URI=$DMLC_PS_ROOT_URI -e DMLC_PS_ROOT_PORT=$DMLC_PS_ROOT_PORT -e DMLC_NUM_WORKER=$DMLC_NUM_WORKER \
 -e DMLC_NUM_SERVER=$DMLC_NUM_SERVER  -e DMLC_ROLE=$DMLC_ROLE -e DMLC_INTERFACE=ib0 \
 -v /export/fanlu/deepspeech_zh_word_dist/:/export/fanlu/deepspeech_zh_word_dist/ \
 -v /export/fanlu/thchs30/:/export/fanlu/thchs30/ -v /export/aiplatform/:/export/aiplatform/ \
 -v /export/fanlu/aishell/:/export/fanlu/aishell/ -w /export/fanlu/deepspeech_zh_word_dist/ \
 dl/python python main.py --configfile deepspeech_dist.cfg
