### Kashgari Workshop

0. prepare GPU instance (AWS/Google Cloud) with drivers and cuDNN
1. run `prep.sh`, get the dependencies and data ready.
2. `python3 classification.py`, train & save model (GPU instance required)
    * p3 2-3 minutes
    * g4dn 5-10 minutes
3. run `docker.sh` to serve the model
4. run `client.py` to test if server is working correctly