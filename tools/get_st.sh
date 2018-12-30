#!/bin/bash

out_path="models/"
link="https://s3-us-west-1.amazonaws.com/yysijie-data/public/st-gcn/models/"
reference_model="resource/reference_model.txt"

mkdir -p $out_path
while IFS='' read -r line || [[ -n "$line" ]]; do
    wget -c $link$line -O $out_path$line
done < "$reference_model"


