import struct
import numpy as np
import io

def find_python_list_format(format_:str):
    if format_=="int8":
        return '<B'
    elif format_=="int16":
        return '<H'
    elif format_=="int32":
        return '<I'
    elif format_=="int64":
        return '<Q'
    elif format_=="float32": 
        return '<f'
    elif format_=="double64":
        return '<d'
 

formats = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.int16,
    np.int8
]


def write_little_endian(dst: io.BytesIO, value: int|float, specific_type: str):
    binary_format = ''
    if isinstance(value,int)==True and specific_type == "uint16_t":
        binary_format = '<H'
    if isinstance(value,int)==True and specific_type == "uint32_t":
        binary_format = '<I'
    if isinstance(value,float)==True and (specific_type == "uint32_t" or specific_type == None): 
        binary_format = '<d'
    
    binary_value = struct.pack(binary_format,value)
    dst.write(binary_value)

def write_fixlen_int(dst: io.BytesIO, val:int):
    binary_value_size = struct.pack('<b',struct.calcsize('i'))
    dst.write(binary_value_size)
     
def write_varlen_int(dst, val):
    if val < 256:
        dst.write(struct.pack('<b',1))
        dst.write(struct.pack('<B',val))
        
    elif val <= 4294967295:
        dst.write(struct.pack('<b',4))
        write_little_endian(dst,val,"uint32_t")
    else:
        dst.write(struct.pack('<b',8))
        write_little_endian(dst,int(val),"uint64_t")
    
def write_sample_values(dst:io.BytesIO, sample:np.ndarray | list, length_of_channels:int , fm : str):
    if isinstance(sample,list):
        if type(sample[0])!=str and type(sample[0])!=list:
            value = b''
            binary_format = find_python_list_format(fm)
            for i in range(length_of_channels):
                value += struct.pack(binary_format,sample[i])
            dst.write(value)
            return sample[length_of_channels:]
        else:
            for i in range(length_of_channels):
                write_varlen_int(dst,len(sample[i]))
                dst.write(bytes(sample[i],'utf-8'))
            return sample[length_of_channels:]   
    
    if isinstance(sample,np.ndarray):
        if sample.dtype in formats:
            binary_value = sample[:length_of_channels].tobytes()
            dst.write(binary_value)
            return sample[length_of_channels:]
        else:
            for i in range(length_of_channels):
                write_varlen_int(dst,len(sample[i]))
                dst.write(bytes(sample[i],'utf-8'))  
            return sample[length_of_channels:]
    
    if isinstance(sample,str):
        write_varlen_int(dst,len(sample))
        dst.write(bytes(sample,'utf-8'))
        return 
        
