CONFIG_DIR=$1
REPEAT=$2
MAX_JOBS=${3:-2}
SLEEP=${4:-1}
MAIN=${5:-main}

(
  trap 'kill 0' SIGINT
  CUR_JOBS=0
  for CONFIG in "$CONFIG_DIR"/*.yaml; do
    if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
      ((CUR_JOBS >= MAX_JOBS)) && wait -n
      python $MAIN.py --cfg $CONFIG --repeat $REPEAT --mark_done &
      echo $CONFIG
      sleep $SLEEP
      ((++CUR_JOBS))
    fi
  done

  wait
)
