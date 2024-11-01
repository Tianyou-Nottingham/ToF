from socket import *
import numpy as np

def TCP_server_connect():
    server_socket = socket(AF_INET, SOCK_STREAM)

    server_socket.bind(('127.0.0.1', 8088))
    server_socket.listen(1)

    print('Waiting for connection...')
    connection_socket, addr = server_socket.accept()
    print('Connected by', addr)
    return connection_socket

def TCP_send(socket, data):
    socket.send(data)

def TCP_receive(socket):
    data = socket.recv(32)
    return data

def exit(socket):
    print('Connection closed')
    socket.close()

if __name__ == "__main__":
    connection_socket = TCP_server_connect()
    print('Connected')
    send = '[924,-45]'
    
    if TCP_receive(connection_socket) == b'OK':
        print('Received OK')
        TCP_send(connection_socket, send.encode())
        print('Sending data...: ', send)
    else:
        print('No data received')
        
    exit(connection_socket)