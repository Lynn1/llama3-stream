# LLAMA3-STREAM

A simple and efficient llama3 local service deployment solution that supports real-time streaming response and is optimized for common Chinese character garbled characters.

# Quick Start

## Step 1 Install the environment

### 1 Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
```

In my test, I used cuda_11.8

### 2 Create a python venv

```bash
conda create -n llama3 python==3.10
conda activate llama3
pip install torch fairscale fire tiktoken==0.4.0 blobfile
```

 (according to [meta-llama/llama3/requirements.txt](https://github.com/meta-llama/llama3/blob/main/requirements.txt))

## Step 2 Download the source code

```bash
git clone https://github.com/Lynn1/llama3-stream
cd llama3-stream
git clone https://github.com/meta-llama/llama3.git
cd llama3
pip install -e .
```

## Step 3 Download the 'original' version of llama3 models

There are two way to download the 'original' version of llama3 models:

Please refer to the official guide to download : [meta-llama/llama3: The official Meta Llama 3 GitHub site](https://github.com/meta-llama/llama3)

If you choose to download them from huggingface, please note that we are using the model format from the original folder
`https://huggingface.co/meta-llama/Meta-Llama-3-*B-Instruct/orignial`

## *Step 4 (Optional) Recut the 70B model shards to fit your local GPU numbers

If you have >=8xGUPs on your computer, or you only want to test the small 7B model, you can skip this step and directly go to Step 5.

Or, you may need to redo the "Horizontal Model Sharding" process to let the large 70B model reason in parallel on multiple GPUs.

I've prepared a convenient script for this step, you can find it here: [Lynn1/llama3-on-2GPUs](https://github.com/Lynn1/llama3-on-2GPUs)

## Step 5 Run and play

Once you have the right model ready, you can run the code to start a server and play with the client.

**Note that:** befor you run, I recommand to comment the 80-81 lines in the llama3/llama/generation.py

```python
#if local_rank > 0:
#    sys.stdout = open(os.devnull, "w")
```

(because later we'll need to verify the http-server process on each rank is ready through terminal output)

**Run the server:**

(Remember to replace the *server ip address* in the `run_server.sh` file to your own ip)

```bash
./run_server.sh
```

Assuming you use MP=2 in Step 4, you will see 2 lines of service ready prompt in the terminal:

' 0: starting http-server xxxx:xx … '

' 1: starting http-server xxxx:xx … '

At this point, you can run the client from another computer:

(Again, remember to replace the *server ip address* in the `run_server.sh` file to your own ip)

```python
python ./stream_client.py
```

You can flirt with it now.
