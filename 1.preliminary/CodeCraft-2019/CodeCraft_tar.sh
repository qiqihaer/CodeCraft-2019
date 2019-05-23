#!/bin/bash

SCRIPT=$(readlink -f "$0")
BASEDIR=$(dirname "$SCRIPT")
cd $BASEDIR

if [ ! -d CodeCraft-2019 ]
then
    echo "ERROR: $BASEDIR is not a valid directory of SDK_python for CodeCraft-2019."
    echo "  Please run this script in a regular directory of SDK_python."
    exit -1
fi

rm -f CodeCraft_code.tar.gz
tar -zcPf CodeCraft_code.tar.gz *