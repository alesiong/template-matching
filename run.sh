#!/usr/bin/env bash
source .env
set -e
ssh $HOST $PROJ_DIR/../clean.sh

echo 'Copying files...'
scp -q -r * $HOST:$PROJ_DIR/

echo 'Compiling...'
ssh $HOST "bash -l -c '\
  cd $PROJ_DIR/; \
  nvcc -O3 -g -o main *.c *.cu --compiler-options -Wall,-Wextra,-Wno-unused-result;\
  '"

echo 'Running...'
ssh $HOST "bash -l -c '\
  cd $PROJ_DIR/; \
  ./main $1 $2 $3; \
  '"
echo 'Copying back results'

mkdir -p output
scp -q -r $HOST:$PROJ_DIR/$3 output/
