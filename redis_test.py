import redis_test
# Connect to Redis
r = redis_test.Redis(host='localhost', port=6379, db=0)

# Test if Redis is responding
try:
    response = r.ping()
    if response:
        print("Redis is working!")
except redis_test.exceptions.ConnectionError:
    print("Failed to connect to Redis.")
