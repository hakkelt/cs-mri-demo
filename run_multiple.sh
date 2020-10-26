#!/bin/sh

for f in $1/*.conf
do
     nohup ./run.sh ${f%.*} > ${f%.*}.log &
done
