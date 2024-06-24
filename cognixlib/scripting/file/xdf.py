"""A module for the creation of an XDF file"""

import io, time, struct, threading
import numpy as np
import xmltodict

from ...scripting.file.conversions import *
from pylsl import local_clock

def write_ts(out:io.StringIO,ts:float,specific_format):
    if (ts==0):
        out.write(struct.pack('<b',0))
    else:
        out.write(struct.pack('<b',8))
        write_little_endian(out,ts,specific_format)

class ChunkTag:
    fileheader = 1
    streamheader = 2
    samples = 3
    clockoffset = 4
    boundary = 5
    streamfooter = 6
    undefined = 0
     
class XDFWriter:
    def __init__(self, filename:str ,on_init_open:bool = False):
        self.filename = filename 
        self.write_mut = threading.Lock()
        self.stream_ids_formats = dict()
        
        if on_init_open:
            self.open_file_(self.filename)
    
    def __enter__(self):
        self.open_file_(self.filename)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close_file()
        
    def open_file_(self, filename: str):
        self._file = open(filename,"wb")
   
        self._file.write("XDF:".encode('utf-8'))
        
        header = "<?xml version=\"1.0\"?>\n  <info>\n    <version>1.0</version>"
        time_now = time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime())
        header += "\n    <datetime>" + time_now + "</datetime>"
        header += "\n  </info>"
        self._write_chunk(ChunkTag.fileheader,header,None)
    
    def close_file(self):
        self._file.close()
        self._file = None

    def write_header(self, streamid: int, stream_id_infos: dict):
        header = ("<?xml version=\"1.0\"?><info><name>{}</name><type>{}</type><channel_count>{}</channel_count><channels>{}</channels><nominal_srate>{}</nominal_srate> \
                <channel_format>{}</channel_format><created_at>{}</created_at></info>".format(stream_id_infos['stream_name'],stream_id_infos['stream_type'],stream_id_infos['channel_count'],\
                    stream_id_infos['channels'],stream_id_infos['nominal_srate'],stream_id_infos['channel_format'],stream_id_infos['time_created']))
        self.write_stream_header(streamid, header)
        self.write_boundary_chunk()
    
    def write_data(
        self,
        streamid: int,
        data_content: list | np.ndarray,
        timestamps: list | np.ndarray,
        channel_count: int=0
    ):
        if isinstance(data_content,list):
            self.write_data_chunk(
                streamid,
                timestamps,
                data_content, 
                channel_count
            )
        else: 
            self.write_data_chunk_nested(
                streamid,
                timestamps,
                data_content
            )
            
    def write_footer(
        self, 
        streamid: int,
        first_time: float,
        last_time: float,
        samples_count: int
    ):
        footer = (
                "<?xml version=\"1.0\"?><info><first_timestamp>{}</first_timestamp><last_timestamp>{}</last_timestamp><sample_count>{}</sample_count> \
                <clock_offsets><offset><time>50979.7660030605</time><value>-3.436503902776167e-06</value></offset></clock_offsets></info>".format(first_time,last_time,samples_count)
            )
        self.write_boundary_chunk()
        self.write_stream_offset(streamid, local_clock(),-0.5)
        self.write_stream_footer(streamid, footer)
    
    def _write_chunk(self,tag:ChunkTag,content:bytes,streamid_p:int):
        self.write_mut.acquire()
        self._write_chunk_header(tag,len(content),streamid_p)
        if isinstance(content,str):
            content = bytes(content,'utf-8')
        self._file.write(content)
        self.write_mut.release()
    
    def write_data_chunk_(self,streamid:int,timestamps:list,chunk:list|np.ndarray,n_samples:int,n_channels:int):
        if n_samples == 0:return 
        if len(timestamps)!=n_samples:
            raise RuntimeError("Timestamp and samples count are mismatched")
        else:##Generate [Samples] chunk contents...
            out = io.BytesIO()
            print(self.stream_ids_formats)
            write_fixlen_int(out,0x0FFFFFFF)  
            for ts in range(len(timestamps)):
                write_ts(out,timestamps[ts],"uint32_t")
                chunk = write_sample_values(out,chunk,n_channels,self.stream_ids_formats[str(streamid)])
            outstr = out.getvalue()
            out.close()
             ##// Replace length placeholder           
            s = struct.pack('<I', n_samples)
            outstr_string_version = struct.pack('b',outstr[0]) + s + outstr[1:]
            self._write_chunk(ChunkTag.samples,outstr_string_version,streamid)
    
    def write_data_chunk(self,streamid:int,timestamps:list,chunk:list|np.ndarray,n_channels:int):
        if type(chunk)!=str and isinstance(chunk,np.ndarray):
            assert len(timestamps) * n_channels == chunk.size
        
        if type(chunk)!=str and isinstance(chunk,list):
            assert len(timestamps) * n_channels == len(chunk)
    
        self.write_data_chunk_(streamid,timestamps,chunk,len(timestamps),n_channels)
        
    def write_data_chunk_nested(self,streamid:int,timestamps:list,chunk:list|np.ndarray): ##### WRITE CHUNK DATA THAT ARE 2D ARRAY
        if isinstance(chunk,list) and len(chunk) == 0:
            return 
        if isinstance(chunk,np.ndarray) and chunk.size == 0: 
            return 
        
        n_samples = len(timestamps)
        if isinstance(chunk, (np.ndarray, list)) and len(timestamps) != chunk.shape[0]:
            raise RuntimeError("Timestamp and sample count are not the same")
        
        if isinstance(chunk, np.ndarray):
            n_channels = chunk.shape[1]
        if isinstance(chunk,list): 
            n_channels = len(chunk[0])
        ## generate [Samples] chunk contents...
        out = io.BytesIO()
        write_fixlen_int(out,0x0FFFFFFF)    
        print(self.stream_ids_formats)  
        for ts in range(len(timestamps)):
            chunk_new = chunk[ts]
            assert(n_channels == len(chunk_new))
            write_ts(out,timestamps[ts],"uint32_t")
            write_sample_values(out,chunk_new,n_channels,self.stream_ids_formats[str(streamid)])
            
        outstr = out.getvalue()
        out.close()
        ##// Replace length placeholder           
        s = struct.pack('<I', n_samples)
        outstr_string_version = struct.pack('b',outstr[0]) + s + outstr[1:]
        self._write_chunk(ChunkTag.samples,outstr_string_version,streamid)
    
    def _write_chunk_header(self,tag:ChunkTag,length:int,streamid_p:int):
        length += struct.calcsize('h')
        if streamid_p!=None:
            length += len(struct.pack('i',streamid_p))
        write_varlen_int(self._file,length)
        write_little_endian(self._file,tag,"uint16_t")
        if streamid_p!=None:
            write_little_endian(self._file,streamid_p,"uint32_t")
        
    def write_stream_header(self, streamid: int, content: str, fm = None):
        if not fm:
            try:
                header_data = xmltodict.parse(content)
                print("HEADER",self.stream_ids_formats)
                self.stream_ids_formats["{}".format(streamid)] = header_data['info']['channel_format']
                print("HEADER",self.stream_ids_formats)
                self._write_chunk(ChunkTag.streamheader,content,streamid)
                
            except Exception as e:
                print("Channel format is missing in xml !!!!!!")
        else:
            print("HEADER",self.stream_ids_formats)
            self.stream_ids_formats["{}".format(streamid)] = fm
            print("HEADER",self.stream_ids_formats)
            self._write_chunk(ChunkTag.streamheader,content,streamid)
           
    def write_stream_footer(self, streamid: int, content: str):
        self._write_chunk(ChunkTag.streamfooter,content,streamid)
    
    def write_stream_offset(self,streamid:int,time_now:time,offset:float):
        self.write_mut.acquire()
        length = struct.calcsize('d') + struct.calcsize('d')
        self._write_chunk_header(ChunkTag.clockoffset,length,streamid)
        write_little_endian(self._file,time_now - offset,None)
        write_little_endian(self._file,offset,None)
        self.write_mut.release()
    
    def write_boundary_chunk(self):
        self.write_mut.acquire()
        boundary_uuid = [0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F, 0xB3, 0x0E,0xD5, 0x46, 0x73, 0x83, 0xCB, 0xE4]
        boundary_uuid = np.array(boundary_uuid,dtype=np.uint8)
        self._write_chunk_header(ChunkTag.boundary,len(boundary_uuid),None)
        self._file.write(boundary_uuid.tobytes())
        self.write_mut.release()

def creation_of_xdf(xdfile:XDFWriter,streamid:int,stream_id_infos:dict,data_content:list|np.ndarray,timestamps:list|np.ndarray,write_header:bool,write_data:bool,write_footer:bool,first_time:float,last_time:float,samples_count:int):
    # print("Creation of XDF FILE for stream {}".format(stream_id_infos['stream_name']))
    # print(stream_id_infos)
    
    if write_header==True:
        # print("Write Header")
        header = ("<?xml version=\"1.0\"?><info><name>{}</name><type>{}</type><channel_count>{}</channel_count><channels>{}</channels><nominal_srate>{}</nominal_srate> \
                <channel_format>{}</channel_format><created_at>{}</created_at></info>".format(stream_id_infos['stream_name'],stream_id_infos['stream_type'],stream_id_infos['channel_count'],\
                    stream_id_infos['channels'],stream_id_infos['nominal_srate'],stream_id_infos['channel_format'],stream_id_infos['time_created']))
        xdfile.write_stream_header(streamid,header)
        xdfile.write_boundary_chunk()
    
    elif write_footer == True:   
        # print("Write Footer")    
        footer = (
                "<?xml version=\"1.0\"?><info><first_timestamp>{}</first_timestamp><last_timestamp>{}</last_timestamp><sample_count>{}</sample_count> \
                <clock_offsets><offset><time>50979.7660030605</time><value>-3.436503902776167e-06</value></offset></clock_offsets></info>".format(first_time,last_time,samples_count)
            )
        xdfile.write_boundary_chunk()
        xdfile.write_stream_offset(streamid, local_clock(),-0.5)
        xdfile.write_stream_footer(streamid,footer)
    
    elif write_data == True:
        if isinstance(data_content,list):
            xdfile.write_data_chunk(streamid,timestamps,data_content,stream_id_infos['channel_count'])
        else: 
            xdfile.write_data_chunk_nested(streamid,timestamps,data_content)
    
        
    
        
        
        
            


        
        
        