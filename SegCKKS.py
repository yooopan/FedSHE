import math
import numpy as np
from Pyfhel import Pyfhel
import json
import warnings
warnings.filterwarnings('ignore')

with open('ModDict.json', 'r') as fcc_file:
    schemeDict = json.load(fcc_file)

def generate_ckks_key(sec_level, mul_depth, poly_moduls_degree):
    HE = Pyfhel()
    ckks_params = schemeDict.get(sec_level, {}).get(mul_depth, {}).get(poly_moduls_degree, {})
    print("ckks_params: ", ckks_params)
    status = HE.contextGen(**ckks_params)
    print(status)
    HE.keyGen()
    return HE


def enc_vector(HE, arr_x):
    ptxt_x = HE.encodeFrac(arr_x)
    ctxt_x = HE.encryptPtxt(ptxt_x)
    return ctxt_x


def dec_vector(HE, ctxt_x):
    r_x = HE.decryptFrac(ctxt_x)
    _r = lambda x: np.round(x, decimals=3)
    return _r(r_x)


def seg_enc_vector(HE, vector, vecl):
    block_enc_arr = [] 
    block_len = HE.get_nSlots()
    block_arr_len = math.ceil(vecl / block_len)
    for i in range(block_arr_len):
        start_index = block_len * i
        end_index = block_len * (i+1)
        if end_index > vecl:
            end_index = vecl
        vector_block = vector[start_index:end_index]
        enc_vector_block = enc_vector(HE, vector_block)
        block_enc_arr.append(enc_vector_block)
    return block_enc_arr


def seg_dec_vector(HE, block_enc_arr):
    dec_result = []
    for block_enc in block_enc_arr:
        dec_result.append(dec_vector(HE, block_enc))
    dec_result = np.concatenate(dec_result)
    return dec_result
