import zipfile
from zipfile import *

def decrypt_and_open(file_name):
    my_password = 'pass123'
    
    with zipfile.ZipFile(file_name) as file:
        file.extractall(pwd=bytes(my_password,'utf-8'))



decrypt_and_open( "/Users/romanzacharek/Desktop/CVTest.zip" )