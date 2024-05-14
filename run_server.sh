

# export CUDA_VISIBLE_DEVICES="0,1,2,3" # 4 GPUs
export CUDA_VISIBLE_DEVICES="2,3" # Correct filling the GPU indexs you want to use

MODEL_FOLDER_PATH="./Meta-Llama-3-70B-Instruct-2shards" # model path
MP=2 # worldsize, model-parallel values

# This is the model parallel process data rendezvous port of 'torchrun': --master_addr 127.0.0.1 --master_port=29500 """ 
# only test stream_generator.py:
# torchrun --nproc_per_node ${MP} stream_generator.py --ckpt_dir ${MODEL_FOLDER_PATH} --tokenizer_path ${MODEL_FOLDER_PATH}/tokenizer.model 

# Start the llama3 server : Stream a response to each request
# Replace the server ip address to your own ip address
torchrun --nproc_per_node ${MP} stream_server.py \
    --ckpt_dir ${MODEL_FOLDER_PATH}/ \
    --tokenizer_path ${MODEL_FOLDER_PATH}/tokenizer.model \
    --max_seq_len 2048 --max_batch_size 1 \
    --server_ip 192.168.1.101 --server_port 8080


