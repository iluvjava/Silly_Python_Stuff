# Only characters with value from 32 to 126 inclusive are allowed.
# range is 91
from getpass import getpass
import re as regex
import os
clear = lambda: os.system("cls")

import random as rd
all = ["encode"]


def FilterString(s):
    filtered = ""
    for C in s:
        if ord(C) <= 126 and ord(C) >= 32:
            filtered += C
    return filtered

def Encode(ps, text, mode=True):
    encoded = ""
    rd.seed(ps)
    for C in text:
        encoded += chr(32 + (rd.randint(0, 94) + (ord(C) - 32))%95) if mode else\
            chr(32 + (-rd.randint(0, 94) + (ord(C) - 32))%95)
    return encoded

def ParseInput(stringInput):
    s = stringInput.index("\"")
    t = stringInput.rindex("\"")
    return stringInput[s + 1:t], stringInput[-1] == "e"

def GetPassWord():
    while True:
        print("Input your password: ")
        ps1 = getpass()
        print("Repeated your password: ")
        ps2 = getpass()
        if ps1 == ps2:
            clear()
            return ps1
        else:
            print("Pass word mismatched, try again please. ")



def main():
    while True:
        text = input("input format: \"<string to encrypt or decrypt>\" [e|d]\n")
        while not regex.match("\\\".+\\\" [ed]", text):
            text = input("Patterns Mismatched, Try again: \n")
            continue
        clear()
        text, mode = ParseInput(text)
        ps = GetPassWord()
        print("This is the encrypted text: ")
        print(Encode(ps, text, mode))
        input()


if __name__ == "__main__":
    main()