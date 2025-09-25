# mqtt_echo_server.py
import argparse
import paho.mqtt.client as mqtt

REQUEST_TOPIC = "bench/echo"
RESPONSE_TOPIC = "bench/echo/response"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to MQTT Broker!")
        client.subscribe(REQUEST_TOPIC)
        print(f"Subscribed to topic: {REQUEST_TOPIC}")
    else:
        print(f"Failed to connect, return code {rc}\n")

def on_message(client, userdata, msg):
    """Echo received message to the response topic."""
    # Simply forward the message payload to the response topic
    client.publish(RESPONSE_TOPIC, msg.payload, qos=0)

def main():
    parser = argparse.ArgumentParser(description="MQTT Echo Server for Benchmark")
    parser.add_argument("--host", default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--ca", help="Path to CA certificate file for TLS")
    parser.add_argument("--cert", help="Path to client certificate file for TLS")
    parser.add_argument("--key", help="Path to client private key file for TLS")
    args = parser.parse_args()

    client = mqtt.Client(client_id="mqtt-echo-server")
    client.on_connect = on_connect
    client.on_message = on_message

    if args.ca:
        print(f"Connecting with TLS (CA: {args.ca})")
        client.tls_set(ca_certs=args.ca, certfile=args.cert, keyfile=args.key)

    try:
        client.connect(args.host, args.port, 60)
    except Exception as e:
        print(f"Error connecting to broker: {e}")
        return

    print(f"MQTT Echo Server started. Listening on {args.host}:{args.port}...")
    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    client.loop_forever()

if __name__ == "__main__":
    main()
