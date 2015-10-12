#!/usr/bin/env bash

if [ ! `which gcc-4.4` ]; then
	if [ ! `which gcc-5` ]; then
		gcc=`which gcc`
	else
		gcc=`which gcc-5`
	fi
else
	gcc=`which gcc-4.4`
fi

if [ ! -d build ]; then
	mkdir build
fi

cd build
rm -rf *
cmake -DCMAKE_C_COMPILER=${gcc} ..
make
cd ..
ln -sf build/fastLDA .


