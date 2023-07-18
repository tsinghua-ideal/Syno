#!/bin/bash
while true
do
    python client.py "$@"
    sleep 60s
done
