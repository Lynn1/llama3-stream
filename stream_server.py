
"""
https://github.com/Lynn1 
update 2024.7.31
Add support for dialog interrupts

update 2024.5.11
This code starts a generate_messages main process that communicates with the http_server child process via pipe,
After the generate_messages process receives the request string, it calls the generator to generate the response string and sends it to the http_server process via pipe stream.
The http_server subprocess listens for external client HTTP requests, parses the client request string, and passes it to the generate_messages process for a response.
"""

from http.server import BaseHTTPRequestHandler, HTTPServer
import multiprocessing
import time
import urllib.parse
import fire
from typing import Optional
import os
from stream_generator import LLMGenerator
import torch
import torch.distributed as dist

MP = 2  #be consistent with run_server.sh
INTERRUPT_SIGNAL = "<INTERRUPT_SIGNAL>"
localrank = 0
parent_conn1 = None # Process communication pipeline: Used by the foreground process (generator) to pass the generated string to the background process
child_conn1 = None  # Process communication pipeline: Used by the background process (generator) to pass the generated string to the foreground process

def check_interrupt(conn,localrank=0):
    interrupt_tensor = torch.zeros(1)
    if localrank<1:
        if conn.poll():  # Check whether there are messages from the http process
            if conn.recv() == INTERRUPT_SIGNAL:
                interrupt_tensor += 1
        for i in range(1,MP):
            dist.send(tensor=interrupt_tensor,dst=i) #Broadcast to others
    else:
        dist.recv(tensor=interrupt_tensor,src=0)

    if interrupt_tensor[0] > 0:
        return True
    return False

def response_stream(generator,conn,prompt,localrank=0):
    res=""
    interrupt=False
    for response in generator.stream_chat(prompt):
        interrupt=check_interrupt(conn,localrank) #Check for interrupt signals in each stream_chat loop
        if interrupt:
            break
        if response != '<user_end>':
            res += response
            if localrank<1:
                conn.send(response) # The resulting string is sent to the http process
    return res,interrupt

def hold_response(generator,conn):
    while True:
        request_str = ""
        request_str = conn.recv()
        if request_str:  # Check for new request strings
            res,interrupt = response_stream(generator,conn,request_str,localrank)
            if interrupt:
                print(f"{localrank} hold_response: interrupt at line 61")
                continue
            # for response in generator.stream_chat(request_str):
            #     if localrank<1:
            #         conn.send(response)# The resulting string is sent to the http child process
            # request_str = "" # After the generation, reset the request string
                        # 对话结束标志
            if localrank<1 and (not interrupt): 
                conn.send('<user_end>')
        else:
            time.sleep(0.5)  # Wait a little to receive a new request string


class StreamingHTTPRequestHandler(BaseHTTPRequestHandler):
    # def do_GET(self):
    def do_POST(self):
        global child_conn1
        if not child_conn1:
            print("child_conn1 is None")
            return
        self.send_conn = child_conn1 # Gets the foreground background process communication pipeline

        # Parse the request string from the URL of the GET request
        # path_str = self.path.strip("/")
        # request_str = urllib.parse.unquote(path_str)  # Decode the URL encoded string

        # Parse the request string from the body of the POST request instead of the URL
        request_str = self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8')

        self.send_conn.send(request_str)  # Sends the request string to the foreground inference process

        if localrank>0:
            return
        
        print(f"{localrank}: http receive request: {request_str}")
        start_time = time.time()

        # Set the response header and status code
        self.send_response(200)
        self.send_header('Content-type', 'text/plain; charset=utf-8') # Ensure that the character set type is declared in the response header
        self.end_headers()
        # streaming
        print(f"{localrank}: http sent response: ")
        while True:
            try:
                if self.send_conn.poll():
                    message = self.send_conn.recv()  # Receives the generated string from the generator process
                    text = f"{message}"  # Fetch string\n
                    if text == "<user_end>":
                        self.wfile.write(b'\n')
                        self.wfile.flush()
                        # print('\n')
                        break
                    self.wfile.write(text.encode('utf-8')) # Output character by character using UTF-8 encoding
                    self.wfile.flush() # Force buffer contents to be written out
                    print(text, end='', flush=True)
            except BrokenPipeError:
                print(">>>>>>>>>>>>>>>>>Client disconnected<<<<<<<<<<<<<<")
                self.send_conn.send(INTERRUPT_SIGNAL)  # Send the interrupt signal to the generator process
                break
        print(f"\n{localrank}: dialog timing: {time.time() - start_time:.2f} seconds.\n")

def http_server(
        send_conn,
        server_ip:str="localhost",
        server_port:int=8080,
        server_class=HTTPServer, 
        handler_class=StreamingHTTPRequestHandler
):
    global child_conn1
    child_conn1 = send_conn
    httpd = server_class((server_ip, server_port), handler_class)

    print(f'{localrank}: Starting http-server {server_ip}:{server_port}...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print(f'{localrank}: Stopping http-server.\n')
        httpd.server_close()


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        server_ip:str="localhost",
        server_port:int=8080,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_seq_len: int = 512,
        max_batch_size: int = 4,
        max_gen_len: Optional[int] = None
):
    # build & load llm
    generator = LLMGenerator(ckpt_dir=ckpt_dir,
                    tokenizer_path=tokenizer_path,
                    temperature=temperature,
                    top_p=top_p,
                    max_seq_len=max_seq_len,
                    max_batch_size=max_batch_size,
                    max_gen_len=max_gen_len)
    
    # Configure the HTTP service IP address, Port, and communication channel
    global localrank
    localrank = int(os.environ.get("LOCAL_RANK", 0)) #LOCAL_RANK: Gets the GPU index of the current parallel inference process

    server_IP = server_ip
    server_Port = server_port+localrank # Each parallel inference process listens on a different port
    global parent_conn1, child_conn1
    parent_conn1, child_conn1 = multiprocessing.Pipe()
    
    # Start the background process - customer request listening service
    print(f"Main {localrank}: Create http-server process on {server_IP}:{server_Port}...")
    http_process = multiprocessing.Process(target=http_server, args=(child_conn1,server_IP,server_Port,))
    http_process.daemon = True
    http_process.start()

    # To start the foreground process loop - call the generator when the request is received
    hold_response(generator,parent_conn1)
   

if __name__ == '__main__':
    fire.Fire(main)