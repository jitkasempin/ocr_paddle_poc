#!/bin/bash
# Script to reload docker compose for gradio3 service

echo "Stopping docker compose..."
docker compose down

echo "Waiting 5 seconds..."
sleep 5

echo "Building gradio3 service..."
docker compose build gradio3

echo "Waiting 5 seconds..."
sleep 5

echo "Starting gradio3 service in detached mode..."
docker compose up -d gradio3

echo "Done! gradio3 service has been reloaded."