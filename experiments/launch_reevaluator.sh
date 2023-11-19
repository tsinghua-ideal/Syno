while true
do
    python reevaluator.py "$@"
    echo "Restarting server..."
    sleep 5s
done
