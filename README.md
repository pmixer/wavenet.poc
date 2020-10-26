KIS(Keep It Short) version of WaveNet, in reference to https://github.com/tomlepaine/fast-wavenet (pls visit their repo for complete documents), coded in PyTorch, to show the concept rather creating a complete repo for experiment or production use,  pls run:

```sh
python model.py
```

for training the model then generate the audio.

Long term goal is to implement an optimized(maybe persistent) CUDA version of WaveNet for model inference in reference to https://github.com/NVIDIA/nv-wavenet/ and try to deploy it using https://github.com/triton-inference-server, pls consider star the repo to let me know KIS version of models are welcomed, so we can create more of them :)
