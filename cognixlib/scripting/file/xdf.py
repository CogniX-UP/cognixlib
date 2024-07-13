"""A module for the creation of an XDF file"""

import io, time, struct, threading
import numpy as np

from collections.abc import Sequence
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from enum import IntEnum
from pylsl import local_clock
from dataclasses import dataclass

from ...scripting.file.conversions import *

from typing import Any

def _write_ts(out:io.StringIO,ts:float,specific_format):
    if (ts==0):
        out.write(struct.pack('<b',0))
    else:
        out.write(struct.pack('<b',8))
        write_little_endian(out,ts,specific_format)

class ChunkTag(IntEnum):
    """Enum Indicating what the -to be written- chunk is."""
    UNDEFINED = 0
    FILE_HEADER = 1
    STREAM_HEADER = 2
    SAMPLES = 3
    CLOCK_OFFSET = 4
    BOUNDARY = 5
    STREAM_FOOTER = 6
    
class XDFHeader:
    
    def __init__(
        self,
        name: str,
        type_: str,
        channels: Sequence[str],
        nominal_srate: int,
        channel_format: str,
        time_created,
        channel_infos: dict[str, Any] = None
    ):
        self.name = name
        self.type = type_
        self.channels = channels
        self.nominal_srate = nominal_srate
        self.channel_format = channel_format
        self.time_created = time_created
        self.channel_infos = channel_infos
        
    @property
    def channel_count(self):
        return len(self.channels)
    
    def to_xml_str(self) -> str:
        info = Element('info')
        
        name = SubElement(info, 'name')
        name.text = self.name
        
        type_ = SubElement(info, 'type')
        type_.text = str(self.type)
        
        ch_count = SubElement(info, 'channel_count')
        ch_count.text = str(self.channel_count)
        
        channels = SubElement(info, 'channels')
        for ch in self.channels:
            channel = SubElement(channels, 'channel')
            label_el = SubElement(channel, 'label')
            label_el.text = str(ch)
            
            if not self.channel_infos:
                continue
            
            for key, val in self.channel_infos.items():
                child = SubElement(channels, key)
                child.text = str(val)
        
        n_srate = SubElement(info, 'nominal_srate')
        n_srate.text = str(self.nominal_srate)
        
        chann_format = SubElement(info, 'channel_format')
        chann_format.text = str(self.channel_format)
        
        created_at = SubElement(info, 'created_at')
        created_at.text = str(self.time_created)
        
        xml_str = ElementTree.tostring(info, encoding='unicode', method='xml')
        return f"<?xml version=\"1.0\"?>{xml_str}"
     
class XDFWriter:
    """A class that allows the writing of data in the XDF format"""
    
    @dataclass
    class _StreamInfo:
        header: XDFHeader = None
        first_time: float = -1.0
        last_time: float = -1.0
        n_samples: int = 0
                 
    def __init__(self, filename: str, on_init_open: bool = False):
        
        self.filename = filename
        if not filename.endswith('.xdf'):
            self.filename = f'{filename}.xdf'
             
        self.write_mut = threading.Lock()
        self._stream_infos: dict[int, XDFWriter._StreamInfo] = {}
        
        if on_init_open:
            self.open_file_(self.filename)
    
    def __enter__(self):
        self.open_file_(self.filename)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close_file()
        
    def open_file_(self, filename: str):
        """Opens the file to be written to."""
        self._file = open(filename,"wb")
   
        self._file.write("XDF:".encode('utf-8'))
        
        header = "<?xml version=\"1.0\"?>\n  <info>\n    <version>1.0</version>"
        time_now = time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime())
        header += "\n    <datetime>" + time_now + "</datetime>"
        header += "\n  </info>"
        self._write_chunk(ChunkTag.FILE_HEADER, None, header)
    
    def close_file(self):
        """Closes the file."""
        for sid in self._stream_infos.keys():
            self._write_footer(sid)
        self._file.close()
        self._file = None

    def add_stream(self, streamid: int, header: XDFHeader):
        """Creates a stream for the XDF."""
        
        xml_str = header.to_xml_str()
        self._stream_infos[streamid] = XDFWriter._StreamInfo(header=header)
        self._write_chunk(ChunkTag.STREAM_HEADER, streamid, xml_str)
        self._write_boundary_chunk()
    
    def write_data(
        self,
        streamid: int,
        data_content: Sequence | np.ndarray,
        timestamps: Sequence[float] | np.ndarray,
    ):
        """Writes the data for a stream for the XDF."""
        
        stream_info = self._stream_infos[streamid]
        if stream_info.first_time < 0:
            stream_info.first_time = timestamps[0]
        stream_info.last_time = timestamps[-1]
         
        self._write_data_chunk(
            streamid,
            timestamps,
            data_content
        )
            
    def _write_footer(
        self, 
        streamid: int,
    ):
        """Writes the footer of a stream for the XDF."""
        
        sinfo = self._stream_infos[streamid]
        first_time = sinfo.first_time
        last_time = sinfo.last_time
        n_samples = sinfo.n_samples
        
        # TODO include clock offsets
        
        info = Element('info')
        first_ts = SubElement(info, 'first_timestamp')
        first_ts.text = str(first_time)
        
        last_ts = SubElement(info, 'last_timestamp')
        last_ts.text = str(last_time)
        
        n_el = SubElement(info, 'sample_count')
        n_el.text = str(n_samples)
        
        xml_str = ElementTree.tostring(info, encoding='unicode', method='xml')
        footer = f"<?xml version=\"1.0\"?>{xml_str}"
        
        #footer = (
        #        f"<?xml version=\"1.0\"?><info><first_timestamp>{first_time}</first_timestamp><last_timestamp>{last_time}</last_timestamp><sample_count>{n_samples}</sample_count></info>"
                # <clock_offsets><offset><time>50979.7660030605</time><value>-3.436503902776167e-06</value></offset></clock_offsets></info>"
        #    )
        self._write_boundary_chunk()
        # Why was this -0.5 instead of 0.0? (if it's 0 an error occurs with struct.pack)
        self._write_stream_offset(streamid, local_clock(), 0.0)
        self._write_chunk(ChunkTag.STREAM_FOOTER, streamid, footer)
    
    def _write_chunk(self,tag: ChunkTag, streamid: int, content: bytes):
        self.write_mut.acquire()
        self._write_chunk_header(tag, streamid, len(content))
        if isinstance(content, str):
            content = bytes(content,'utf-8')
        self._file.write(content)
        self.write_mut.release()
        
    def _write_data_chunk(
        self,
        streamid: int,
        timestamps: Sequence[float],
        chunk: Sequence | np.ndarray
    ):  
        
        if len(chunk) == 0:
            return

        if isinstance(chunk, np.ndarray):
            n_samples, n_channels = chunk.shape
        elif isinstance(chunk, Sequence): 
            n_samples, n_channels = len(chunk), len(chunk[0])

        if len(timestamps) != n_samples:
            raise RuntimeError("Timestamp and sample count are not the same")
        
        self._stream_infos[streamid].n_samples += n_samples
        ## Generate [Samples] chunk contents...
        out = io.BytesIO()
        write_fixlen_int(out, 0x0FFFFFFF)    
        for i in range(len(timestamps)):
            chunk_new = chunk[i]
            assert(n_channels == len(chunk_new))
            _write_ts(out, timestamps[i], "uint32_t")
            fm = self._stream_infos[streamid].header.channel_format
            write_chunk_values(out, chunk_new, n_channels, fm)    
        out_bytes = out.getvalue()
        out.close()
        
        ## Replace length placeholder           
        s = struct.pack('<I', n_samples)
        out_str = struct.pack('b', out_bytes[0]) + s + out_bytes[1:]
        self._write_chunk(ChunkTag.SAMPLES, streamid, out_str)
    
    def _write_chunk_header(self, tag: ChunkTag, streamid_p: int, length: int):
        length += struct.calcsize('h')
        if streamid_p is not None:
            length += len(struct.pack('i', streamid_p))
        write_varlen_int(self._file,length)
        write_little_endian(self._file, tag, "uint16_t")
        if streamid_p is not None:
            write_little_endian(self._file, streamid_p, "uint32_t")
    
    def _write_stream_offset(self, streamid: int, time_now: float, offset: float):
        self.write_mut.acquire()
        length = 2 * struct.calcsize('d')
        self._write_chunk_header(ChunkTag.CLOCK_OFFSET, streamid, length)
        write_little_endian(self._file, time_now - offset, None)
        write_little_endian(self._file, offset, None)
        self.write_mut.release()
    
    def _write_boundary_chunk(self):
        self.write_mut.acquire()
        boundary_uuid = [0x43, 0xA5, 0x46, 0xDC, 0xCB, 0xF5, 0x41, 0x0F, 0xB3, 0x0E,0xD5, 0x46, 0x73, 0x83, 0xCB, 0xE4]
        boundary_uuid = np.array(boundary_uuid, dtype=np.uint8)
        self._write_chunk_header(ChunkTag.BOUNDARY, None, len(boundary_uuid))
        self._file.write(boundary_uuid.tobytes())
        self.write_mut.release()
    
        
    
        
        
        
            


        
        
        