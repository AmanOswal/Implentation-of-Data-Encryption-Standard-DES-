#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
def param(hw):
    SHIFT = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1,1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]
    S_BOX = [

        [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
         [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
         [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
         [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
        ],

        [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
         [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
         [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
         [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
        ],

        [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
         [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
         [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
         [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
        ],

        [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
         [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
         [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
         [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
        ],  

        [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
         [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
         [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
         [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
        ], 

        [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
         [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
         [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
         [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
        ], 

        [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
         [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
         [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
         [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
        ],

        [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
         [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
         [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
         [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
        ],
        [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
         [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
         [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
         [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
        ],

        [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
         [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
         [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
         [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
        ],

        [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
         [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
         [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
         [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
        ],

        [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
         [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
         [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
         [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
        ],  

        [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
         [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
         [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
         [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
        ], 

        [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
         [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
         [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
         [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
        ], 

        [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
         [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
         [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
         [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
        ],

        [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
         [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
         [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
         [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
        ]
        ]
#     print("Input half width") 
#     hw= int(input())
    #Cipher Text always length 8
    #Initial permutation matrix for the datas(Sabse pehle jo 64 bit initial plaintext  par lagega)(Initial Permutation)
    if(hw == 32):
        PI = [58, 50, 42, 34, 26, 18, 10, 2,
              60, 52, 44, 36, 28, 20, 12, 4,
              62, 54, 46, 38, 30, 22, 14, 6,
              64, 56, 48, 40, 32, 24, 16, 8,
              57, 49, 41, 33, 25, 17, 9, 1,
              59, 51, 43, 35, 27, 19, 11, 3,
              61, 53, 45, 37, 29, 21, 13, 5,
              63, 55, 47, 39, 31, 23, 15, 7]

        #Initial permut made on the key(Sabse pehle jo 64 bit initial key  par lagega)(Permuted Choice 1)
        CP_1 = [57, 49, 41, 33, 25, 17, 9,
                1, 58, 50, 42, 34, 26, 18,
                10, 2, 59, 51, 43, 35, 27,
                19, 11, 3, 60, 52, 44, 36,
                63, 55, 47, 39, 31, 23, 15,
                7, 62, 54, 46, 38, 30, 22,
                14, 6, 61, 53, 45, 37, 29,
                21, 13, 5, 28, 20, 12, 4]

        #Permut applied on shifted key to get Ki+1(Permuted Choice 2)
        CP_2 = [14, 17, 11, 24, 1, 5, 3, 28,
                15, 6, 21, 10, 23, 19, 12, 4,
                26, 8, 16, 7, 27, 20, 13, 2,
                41, 52, 31, 37, 47, 55, 30, 40,
                51, 45, 33, 48, 44, 49, 39, 56,
                34, 53, 46, 42, 50, 36, 29, 32]

        #Expand matrix to get a 48bits matrix of datas to apply the xor with Ki(Expansion Permutation)
        E = [32, 1, 2, 3, 4, 5,
             4, 5, 6, 7, 8, 9,
             8, 9, 10, 11, 12, 13,
             12, 13, 14, 15, 16, 17,
             16, 17, 18, 19, 20, 21,
             20, 21, 22, 23, 24, 25,
             24, 25, 26, 27, 28, 29,
             28, 29, 30, 31, 32, 1]

        #Permut made after each SBox substitution for each round(Permutation P(Last Permutation))
        P = [16, 7, 20, 21, 29, 12, 28, 17,
             1, 15, 23, 26, 5, 18, 31, 10,
             2, 8, 24, 14, 32, 27, 3, 9,
             19, 13, 30, 6, 22, 11, 4, 25]

        #Final permut for datas after the 16 rounds(Inverse Initial Permutation)
        PI_1 = [40, 8, 48, 16, 56, 24, 64, 32,
                39, 7, 47, 15, 55, 23, 63, 31,
                38, 6, 46, 14, 54, 22, 62, 30,
                37, 5, 45, 13, 53, 21, 61, 29,
                36, 4, 44, 12, 52, 20, 60, 28,
                35, 3, 43, 11, 51, 19, 59, 27,
                34, 2, 42, 10, 50, 18, 58, 26,
                33, 1, 41, 9, 49, 17, 57, 25]

        #Matrix that determine the shift for each round of keys

    if(hw == 16):
        PI=[27, 18, 13, 21, 19, 7, 28, 31, 26, 15, 32, 24, 3, 4, 22, 11, 1, 30, 5, 6, 25, 12, 20, 16, 10, 14, 17, 8, 9, 23, 29, 2]#32
        E = [16, 1, 2, 3, 4, 5,
         4, 5, 6, 7, 8, 9,
         8, 9, 10, 11, 12, 13,
         12, 13, 14, 15, 16, 1]
        CP_1 = [12,5,6,19,20,3,30,4,26,25,31,22,23,18,21,15,14,28,1,13,27,17,10,7,11,9,2,29]
        CP_2=[24,1,20,10,4,3,6,25,26,22,21,14,28,18,15,7,16,19,17,27,12,23,8,5]
        P=[12,3,4,2,16,1,8,9,14,10,6,15,7,5,13,11]
        PI_1=[17, 32, 13, 14, 19, 20, 6, 28, 29, 25, 16, 22, 3, 26, 10, 24, 27, 2, 5, 23, 4, 15, 30, 12, 21, 9, 1, 7, 31, 18, 8, 11]

    if(hw == 64):
        PI=[32,   52   , 5   ,81  , 26,   38,  114  , 46  , 69  , 16  ,  4 ,  42 , 122 ,  83  , 97   ,95   ,66   ,61,  123 ,  77  , 65  ,117  , 18   ,57  ,  7 ,  22  , 37 ,  70 ,   6,   67,   86 , 120  , 60  ,128 ,  44 ,  90 ,  92 ,   8 , 104 ,  25 , 107 ,  10,  111 , 116 ,  33 , 127  ,126  ,119  ,124  , 54 , 102  ,  1 ,  49  , 27 ,  45  , 30 , 108  , 55  , 13  ,100 ,  94 , 105 , 103 ,  72 ,  99   ,79  ,112 ,  56  , 28   ,24  , 71   ,84  , 78  , 91  ,110  ,113  , 80  , 64  , 29   ,76  ,118   ,34   ,19  , 17   ,20   ,88  ,109   ,47   ,89   ,58  , 23  , 41  , 59   ,62   , 2  , 63  , 15  ,121  , 35   , 3  , 98 , 125  , 75,  101 ,  73  , 96  , 14   ,36   ,50,   85,   43  , 82  , 21  ,106  ,  9  , 48  , 39  , 11   ,68  , 12,   87 ,  93 ,  40  , 31,  115   ,74  , 51  , 53]
        #E=[8,37,64,32,17,32,17,43,40,20,17,23,47,52,8,49,60,40,31,21,16,20,33,14,25,56,49,44,47,41,30,46,12,19,64,18,50,1,18,22,15,55,15,57,11,17,42,51,55,52,28,57,30,20,55,8,8,37,53,9,52,59,63,39,38,51,61,38,45,9,45,37,14,5,24,2,2,54,27,60,42,30,46,21,25,1,32,21,25,42,11,20,20,12,12,11]
        E = [64,1,2,3,4,5,
            4,5,6,7,8,9,
            8,9,10,11,12,13,
            12,13,14,15,16,17,
            16,17,18,19,20,21,
            20,21,22,23,24,25,
            24,25,26,27,28,29,
            28,29,30,31,32,33,
            32,33,34,35,36,37,
            36,37,38,39,40,41,
            40,41,42,43,44,45,
            44,45,46,47,48,49,
            48,49,50,51,52,53,
            52,53,54,55,56,57,
            56,57,58,59,60,61,
            60,61,62,63,64,1]
        CP_1 = [104,42,77,44,20,51,117,39,83,5,17,73,126,114,124,88,99,65,102,2,9,82,70,12,80,121,119,22,105,19,66,26,33,24,3,95,63,98,79,91,47,55,40,90,34,21,115,12,112,81,108,7,69,123,49,94,61,4,27,103,67,18,84,15,36,46,48,13,35,16,50,122,14,10,76,37,29,110,32,74,28,52,71,30,72,11,6,31,56,128,62,120,92,111,25,1,64,125,85,68,59,38,96,97,109,100,101,23,118,60,87,107]
        CP_2 = [45,44,14,3,89,95,64,8,4,16,6,99,48,68,17,7,82,98,55,12,72,61,43,58,47,31,93,24,104,62,108,23,101,88,63,11,27,21,56,22,18,33,100,10,102,36,97,35,109,2,84,81,60,73,92,52,37,28,38,54,5,69,34,50,80,96,70,71,20,77,112,65,78,76,30,106,111,91,59,110,74,46,87,67,32,9,75,1,103,15,90,41,107,39,42,19]
        P = [8,19,14,58,47,63,42,17,25,28,39,4,46,32,44,57,41,33,48,18,15,51,50,6,24,38,40,1,52,2,60,54,45,29,13,23,27,34,7,21,30,3,36,12,31,37,55,20,56,26,61,49,53,10,43,16,22,11,9,35,59,62,64,5]
        PI_1=[52 ,95 ,100 ,11 ,3 ,29 ,25 ,38 ,115 ,42 ,118 ,120 ,59 ,107 ,97, 10 ,84 ,23 ,83 ,85 ,113 ,26, 91 ,70 ,40, 5 ,54 ,69 ,79 ,56, 124, 1, 45, 82, 99, 108 ,27, 6 ,117 ,123 ,92 ,12, 111, 35, 55 ,8, 88 ,116, 53 ,109 ,127 ,2 ,128, 50 ,58 ,68 ,24 ,90 ,93 ,33 ,18 ,94, 96 ,78 ,21 ,17 ,30 ,119, 9, 28 ,71, 64 ,105 ,126 ,103 ,80 ,20, 73, 66, 77 ,4 ,112 ,14, 72, 110 ,31 ,121 ,86 ,89, 36 ,74 ,37, 122, 61 ,16 ,106 ,15 ,101, 65 ,60, 104 ,51, 63 ,39 ,62 ,
          114, 41, 57, 87 ,75 ,43 ,67, 76, 7 ,125, 44, 22, 81, 48 ,32, 98, 13 ,19, 49 ,102 ,47 ,46 ,34 ]

    return (S_BOX, PI,E,CP_1,CP_2,P,PI_1, SHIFT)
    
    


# In[34]:


avae = []
avad = []
ke = []
def string_to_bit_array(text):#Convert a string into a list of bits
    array = list()
    for char in text:
        binval = binvalue(char, 8)#Get the char value on one byte
        array.extend([int(x) for x in list(binval)]) #Add the bits to the final list
    return array

def bit_array_to_string(array): #Recreate the string from the bit array
    res = ''.join([chr(int(y,2)) for y in [''.join([str(x) for x in _bytes]) for _bytes in  nsplit(array,8)]])   
    return res

def binvalue(val, bitsize): #Return the binary value as a string of the given size 
    binval = bin(val)[2:] if isinstance(val, int) else bin(ord(val))[2:]#number-find binary value,if char then of its ascii value
    if len(binval) > bitsize:
        raise "binary value larger than the expected size"
    while len(binval) < bitsize:
        binval = "0"+binval #Add as many 0 as needed to get the wanted size
    return binval

def nsplit(s, n):#Split a list into sublists of size "n"
    return [s[k:k+n] for k in range(0, len(s), n)]

ENCRYPT=1
DECRYPT=0
avae = []
class des():
    def __init__(self):
        self.password = None
        self.text = None
        self.keys = list()
        
    def run(self, key, text, n_rounds,hw, action=ENCRYPT,padding=False):
        if len(key) < (hw//4):
            raise "Key Should be bytes long"
        elif (len(key) > (hw//4)):
            key = key[:(hw//4)] #If key size is above 8bytes, cut to be 8bytes long
        
        self.password = key
        self.text = text
        self.hw = hw
        
        if (padding and action==ENCRYPT):
            self.addPadding(hw)
        elif (len(self.text) % ((self.hw)//4) != 0):#If not padding specified data size must be multiple of 8 bytes
            raise ("Data size should be multiple of 8")
        
        self.generatekeys(n_rounds) #Generate all the keys
        text_blocks = nsplit(self.text, (hw//4)) #Split the text in blocks of 8 bytes so 64 bits
        result = list()
        for block in text_blocks:#Loop over all the blocks of data
            block = string_to_bit_array(block)#Convert the block in bit array
            block = self.permut(block,PI)#Apply the initial permutation
            g, d = nsplit(block, self.hw) #g(LEFT), d(RIGHT)
            tmp = None
            for i in range(n_rounds): #Do the 16 rounds
                d_e = self.expand(d, E) #Expand d to match Ki size (48bits)
                if action == ENCRYPT:
                    tmp = self.xor(self.keys[i], d_e)#If encrypt use Ki
                else:
                    tmp = self.xor(self.keys[n_rounds-1-i], d_e)#If decrypt start by the last key
                tmp = self.substitute(tmp) #Method that will apply the SBOXes
                tmp = self.permut(tmp, P)
                tmp = self.xor(g, tmp)
                g = d
                d = tmp
                if(action == ENCRYPT):
#                     print(i)
                    avae.append(d+g)
                else:
                    avad.append(d+g)
            result += self.permut(d+g, PI_1) #Do the last permut and append the result to result
        final_res = bit_array_to_string(result)
        if padding and action==DECRYPT:
            return self.removePadding(final_res) #Remove the padding if decrypt and padding is true
        else:
            return final_res #Return the final string of data ciphered/deciphered
    
    def substitute(self, d_e):#Substitute bytes using SBOX
        subblocks = nsplit(d_e, 6)#Split bit array into sublist of 6 bits
        result = list()
        for i in range(len(subblocks)): #For all the sublists
            block = subblocks[i]
            row = int(str(block[0])+str(block[5]),2)#Get the row with the first and last bit
            column = int(''.join([str(x) for x in block[1:][:-1]]),2) #Column is the 2,3,4,5th bits
#             print(i,row,column)
            val = S_BOX[i][row][column] #Take the value in the SBOX appropriated for the round (i)
            bina = binvalue(val, 4)#Convert the value to binary
            result += [int(x) for x in bina]#And append it to the resulting list
        return result
        
    def permut(self, block, table):#Permut the given block using the given table (so generic method)
        return [block[x-1] for x in table]
    
    def expand(self, block, table):#Do the exact same thing than permut but for more clarity has been renamed
        return [block[x-1] for x in table]
    
    def xor(self, t1, t2):#Apply a xor and return the resulting list
        return [x^y for x,y in zip(t1,t2)]
    
    def generatekeys(self,n_rounds):#Algorithm that generates all the keys
        self.keys = []
#         print(self.password)
        key = string_to_bit_array(self.password)
#         print(key)
        key = self.permut(key, CP_1) #Apply the initial permut on the key
        g, d = nsplit(key, (len(key))//2) #Split it in to (g->LEFT),(d->RIGHT)
        for i in range(0,n_rounds):#Apply the 16 rounds
#             print(i)
            g, d = self.shift(g, d, SHIFT[i]) #Apply the shift associated with the round (not always 1)
            tmp = g + d #Merge them
            self.keys.append(self.permut(tmp, CP_2)) #Apply the permut to get the Ki
        ke.append(self.keys)
        
    def shift(self, g, d, n): #Shift a list of the given value
        return (g[n:] + g[:n], d[n:] + d[:n])
    
    def addPadding(self, hw):#Add padding to the datas using PKCS5 spec.
        pad_len = (hw//4) - (len(self.text) % (hw//4))
        self.text += pad_len * chr(pad_len)
    
    def removePadding(self, data):#Remove the padding of the plain text (it assume there is padding)
        pad_len = ord(data[-1])
        return data[:-pad_len]
    
    def encrypt(self, key, text,n_rounds, hw,padding=False):
        return self.run(key, text, n_rounds,hw, ENCRYPT, padding)
    
    def decrypt(self, key, text,n_rounds,hw, padding=False):
        return self.run(key, text, n_rounds,hw, DECRYPT, padding)
    

d = des()

#Uncomment till for avalanche for cipher properties

# # For cipher Properties
# hwarr = [16,32,64]
# nrarr = [1,8,16,32]
# key = "secret_kehbhjnuh"
# text= "Hello woHello wohh"
# for hw in hwarr:
#     for n_rounds in nrarr:
#         S_BOX, PI,E,CP_1,CP_2,P,PI_1,SHIFT = param(hw)
#         r = d.encrypt(key,text,n_rounds,hw,padding=True)
#         r2 = d.decrypt(key,r,n_rounds,hw,padding=True) 
#         print("n_rounds=", n_rounds)
#         print("half_width", hw)
#         print("Ciphered: %r" % r)
#         print("Deciphered: ", r2)
#         print()

#For Avalanche:

hw = int(input())
n_rounds = int(input())
# text= "Hello woHello wohh"
key2  = "secret_u"
key1 = "secret_k"
# # # text2= "Hpllo woHello wohh"
# # key1 = "129442149690388956"
# # key2 = "57384555652461020"
# # text = "7531478153206098852"
# d = des()
S_BOX, PI,E,CP_1,CP_2,P,PI_1,SHIFT = param(hw)

# # # d.encrypt(key,text,n_rounds,hw,padding=True)
# # # avafe1 = avae.copy()
# # # avae = []

print(d.encrypt(key1,text,n_rounds,hw,padding=True))
avafe2 = avae.copy()
avae = []
avafd2 = avad.copy()
avad = []

print(d.encrypt(key2,text,n_rounds,hw,padding=True))
avafe3 = avae.copy()
avae = []
avafd3 = avad.copy()
avad = []
#r2 = d.decrypt(key,r,n_rounds,hw,padding=True) 


# # # S_BOX, PI,E,CP_1,CP_2,P,PI_1 = param(16)
# # # if(n_ro)
# # # r = d.encrypt(key,text,32,16,padding=True)
# # # r2 = d.decrypt(key,r,32,16,padding=True)
# # # print("Ciphered: %r" % r)
# # # print("Deciphered: ", r2)


# In[35]:


avafe2 = np.array(avafe2)
avafe3 = np.array(avafe3)
differinge= np.sum(abs(avafe2 - avafe3), axis=1)
print(differinge)


# In[36]:


avafe3.shape


# In[37]:


avafd2 = np.array(avafe2)
avafd3 = np.array(avafe3)
differingd= np.sum(abs(avafd2 - avafd3), axis=1)
print(differingd)


# In[38]:


import matplotlib.pyplot as plt
plt.plot(differingd)
plt.title("DecryptionAvalanche")
plt.show()


# In[39]:


import matplotlib.pyplot as plt
plt.plot(differinge)
plt.title("EncryptionAvalanche")
plt.show()


# In[ ]:




