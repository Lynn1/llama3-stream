
"""
https://github.com/Lynn1 update 2024.4.26 
This code implements an HTTP client that will send a GET request to the HTTP server,
The request contains a string (simulating user input),
The server response is then streamed character by character.
"""

from http.client import HTTPConnection
import urllib.parse
import time


def request_example(
        request_str:str, 
        server_ip:str, 
        server_port:int):
    conn = HTTPConnection(server_ip, server_port,timeout=300)
    conn2 = HTTPConnection(server_ip, server_port+1,timeout=300) # Assuming you use MP=2 in Step 4

    # The request string contains special characters such as Chinese characters
    # The string is urL-encoded using urllib.parse.quote
    encoded_request_str = urllib.parse.quote(request_str)
    # Adds an encoded string to the requested URL
    # time.sleep(0.5)
    request_path = "/" + encoded_request_str
    conn.request("GET", request_path)
    conn2.request("GET", request_path)
    print(f"Send request: {request_str}")

    response = conn.getresponse()
    print(f"Response status: {response.status}")
    print(f"Response reason: {response.reason}")

    # Stream character by character reading (in order to parse UTF-8 characters correctly, a buffer is used to store possible multi-byte characters)
    buffer = bytes()
    while chunk := response.read(1):
        buffer += chunk  # Adds the read bytes to the buffer
        try:
            # Attempt to decode the entire buffer contents
            text = buffer.decode('utf-8')
            print(text, end='', flush=True)
            buffer = bytes()  # Empty the buffer and wait for new content
        except UnicodeDecodeError:
            # If the decoding fails (possibly because a split multibyte character is encountered), more data is read
            continue
    conn.close()
    conn2.close()


if __name__ == '__main__':
    request_str = "怎么从纽约去北京?" # How to get to Beijing from New York? (test with chinese)
    request_example(request_str=request_str, server_ip="192.168.1.101", server_port=8080)  ###replace the ip address of the server side
    time.sleep(2)

    request_str = "请讲个有趣的中文古诗给我听。"
    request_example(request_str=request_str, server_ip="192.168.1.101", server_port=8080) 
    time.sleep(2)

    