#!/usr/bin/env bash

mkdir models
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
mv lid.176.bin /models

curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start