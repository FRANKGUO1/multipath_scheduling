#!/bin/bash

sudo pkill -9 -f 'iperf'
sudo pkill -9 -f 'int_'

# 清空INT数据
sudo echo -n > s1_int.csv
sudo echo -n > s2_int.csv
sudo echo -n > s3_int.csv