#!/bin/bash

WATCH_JOB=57986004
CANCEL_JOB=57995469

while true; do
    STATE=$(squeue -h -j "$WATCH_JOB" -o "%t")

    if [[ "$STATE" == "CF" || "$STATE" == "R" ]]; then
        echo "$(date): $WATCH_JOB state=$STATE, cancelling $CANCEL_JOB"
        scancel "$CANCEL_JOB"
        exit 0
    fi

    if [[ -z "$STATE" ]]; then
        echo "$(date): $WATCH_JOB disappeared from squeue"
        exit 1
    fi

    sleep 5
done
