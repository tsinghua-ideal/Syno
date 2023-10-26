while true
do
    python server.py "$@"
    echo "Restarting server..."
    sleep 5s
done
