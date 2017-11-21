#!/usr/bin/env bash
source .env
set -e
ssh $HOST $PROJ_DIR/../clean.sh

echo 'Copying files...'
scp -q -r * $HOST:$PROJ_DIR/

echo 'Compiling...'
ssh $HOST "bash -l -c '\
  cd $PROJ_DIR/; \
  nvcc --compiler-options -Wall -g -o main *.c *.cu;\
  '"

echo 'Running...'
ssh $HOST "bash -l -c '\
  cd $PROJ_DIR/; \
  mv main data; \
  cd data; \
  ./main; \
  '"
echo 'Copying back results'

mkdir -p output
scp -q -r $HOST:$PROJ_DIR/data/output.bmp output/
