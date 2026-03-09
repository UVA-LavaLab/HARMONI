#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs traces results

rm -rf outputs/*
rm -rf traces/*
rm -rf results/*

#[Optional] (only use this if you change task_mapping in HARMONI)
#rm -rf graph_cache/*
