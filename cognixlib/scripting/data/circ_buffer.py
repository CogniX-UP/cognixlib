import numpy as np
from collections.abc import Sequence

_default_dts_err_scale=2

# TODO Fifx this class to not need to transpose the data before it enters
class CircularBuffer:
    """
    An implementation of a circular buffer for handling data and timestamps
    
    This buffer was made with data x times format, but since then, we have
    decided on a times x data format. Hence, this might need to be changed.
    """
    
    @classmethod
    def create(cls, data: np.ndarray, timestamps: np.ndarray):
        result = CircularBuffer(1, 1, 1, 1, False)
        result.reset(data, timestamps)
        return result
    
    @classmethod
    def empty(cls):
        return CircularBuffer(1, 1, 1, 1, False)
        
    def __init__(
        self,
        sampling_frequency: float, 
        buffer_duration: float, 
        start_time: float,
        channels_count: int,
        init_buffers=True
    ):
        self.nominal_srate = sampling_frequency
        self.effective_srate = 0
        self.duration = buffer_duration
        self.current_index = 0
        
        self.tloop = start_time
        self.dts = 1 / self.nominal_srate

        if init_buffers:
            size = int(buffer_duration * self.nominal_srate)
            self.data = np.full((channels_count, size), -1.0, dtype=float)
            self.timestamps = np.full(size, -1.0, dtype=float)
        
        # for calculating effective srate
        self.interval_count = 0
        self.total_intervals = 0
    
    def reset(self, data: np.ndarray, timestamps: np.ndarray):
        self.timestamps = timestamps
        self.duration = timestamps[-1] - timestamps[0]
        self.tloop = timestamps[-1]
        self.current_index = self.size-1
        self.effective_srate = len(timestamps)/(timestamps[-1] - timestamps[0])
        self.data = data.T
    
    @property
    def size(self):
        return len(self.timestamps)
    
    @property
    def effective_dts(self):
        if self.effective_srate == 0:
            return 0
        return 1 / self.effective_srate
    
    @property
    def current_time(self):
        return self.timestamps[self.current_index]
    
    def append_expand(
        self,
        data: np.ndarray,
        timestamps: Sequence[float],
    ) -> bool:
        """
        Expands the current buffer with the data if the incoming data 
        duration exceeds that of the buffer, else simply appends it.
        
        Returns True if the buffer has expanded.
        """
        # might be a single timestamp
        
        data_dur = (
            timestamps[0] if len(timestamps) == 0
            else timestamps[-1] - timestamps[0]
        )
        
        if data_dur > self.duration:
            data = data.T
            added_dur = data_dur - self.duration
            self.duration += added_dur
            self.data = np.concatenate(self.data, data)
            self.timestamps = np.concatenate(self.timestamps, timestamps)
            self.current_index = self.size - 1
            self.effective_srate = (
                len(self.timestamps) / 
                (self.timestamps[-1] - self.timestamps[0])
            )
            return True
        else:
            self.append(data, timestamps)
            return False
        
    def append(
        self, 
        data: np.ndarray, 
        timestamps: Sequence[float],
        get_looped_data=False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Appends data and corresponding timestamps to the buffer
        
        When get_looped_data=True, it returns a tuple of (data, timestamps) 
        if the buffer has looped, else None.
        """
        
        data = data.T
        
        ts_len = len(timestamps)
        assert data.shape[1] == ts_len, f"Length of data and timestamps was not equal! {data.shape[1]} != {ts_len}"
        
        # TODO validate this later
        # Attempting to remove old timestamps influence
        
        # exponential moving average
        current_erate = len(timestamps) / (timestamps[-1] - timestamps[0])
        if self.effective_srate == 0:
            self.effective_srate = current_erate
        else:
            self.effective_srate = self.effective_srate + 0.65*(current_erate - self.effective_srate)
         
        if self.current_index + ts_len <= self.size:
            self.timestamps[self.current_index :ts_len + self.current_index] = timestamps
            self.data[:, self.current_index:ts_len + self.current_index] = data
            self.current_index += ts_len - 1
            return (None, None)
        
        else:
            extra = ts_len - (self.size - self.current_index)
            breakpoint = ts_len - extra
            
            self.timestamps[self.current_index:self.size] = timestamps[:breakpoint]
            self.data[:,self.current_index:self.size] = data[:, :breakpoint]
            
            self.timestamps[0:extra] = timestamps[breakpoint:]
            self.data[:, 0:extra] = data[:, breakpoint:]
            
            self.tloop = timestamps[breakpoint]

            looped_result = (None, None)
            # might not be needed
            if breakpoint != 0:
                #loop result
                looped_result = (
                    (self.data.copy(), self.timestamps.copy())
                    if get_looped_data else (None, None)
                )

            self.current_index = extra - 1
            
            return looped_result
    
    def find_index(
        self, 
        timestamp: float, 
        error_margin=0.0,
        dts_error_scale=_default_dts_err_scale,
    ):
        """Finds closest index of the buffer based on a timestamp"""

        tc = self.timestamps[self.current_index]
        tx = timestamp
        
        if tx > tc or tc - tx > self.duration:
            tx -= error_margin
         
        if (tx > tc) or ((tc - tx) > self.duration): 
            return (-1, False)

        tl = self.timestamps[0]
        index = int((tx - tl) * self.effective_srate)
        overflow = False
        if index < 0:
            index = self.size - (-index)
            overflow = True
        elif index > self.current_index:
            index = self.current_index
        
        if index >= self.size:
            index = self.size-1
        
        found_time = self.timestamps[index]
        err = found_time - timestamp
        # error is based around effective dts
        dts_err_margin = dts_error_scale * self.effective_dts
        
        if abs(err) <= dts_err_margin:
            return index, overflow
        
        # At this point, we're pretty close to the time we want
        # but still far off from good precision. Hence, we're
        # attempting a local search close to the found index 
        # to narrow down a better index
        search_boundary = int(err / self.effective_dts)
        if search_boundary > 0:
            if index >= search_boundary or index + search_boundary < self.size:
                search_arr = self.timestamps[index-search_boundary:index]
                search_diff = np.abs(search_arr - timestamp)
                found_index = np.argmax(search_diff <= dts_err_margin)
                if search_diff[found_index] <= dts_err_margin:
                    index = index - search_boundary + found_index
                    if index > self.current_index:
                        overflow = True
            else:
                leftover = index - search_boundary
                begin = max(self.size - leftover, self.current_index)
                search_arr = np.concatenate(
                    self.timestamps[begin:self.size],
                    self.timestamps[0, index]
                )
                search_diff = np.abs(search_arr - timestamp)
                found_index = np.argmax(search_diff <= dts_err_margin)
                if search_diff[found_index] <= dts_err_margin:
                    if found_index + begin < self.size:
                       index = found_index
                       overflow = True
                    else:
                       index = self.size - begin + found_index
        
        elif search_boundary < 0:
            search_boundary = abs(search_boundary)
            if index <= self.current_index or index + search_boundary < self.size:
                if index <= self.current_index:
                    search_boundary = min(search_boundary, self.current_index + 1)
                else:
                    search_boundary = min(search_boundary, self.size-1)
                    
                search_arr = self.timestamps[index:index+search_boundary]
                search_diff = np.abs(search_arr - timestamp)
                found_index = np.argmax(search_diff <= dts_err_margin)
                if search_diff[found_index] <= dts_err_margin:
                    index += found_index
                    if index > self.current_index:
                        overflow = True
                        
            elif index + search_boundary >= self.size:
                search_boundary = max(search_boundary - (self.size - index), 0)
                search_arr = np.concatenate(
                    self.timestamps[index:self.size],
                    self.timestamps[0:search_boundary]
                )
                search_diff = np.abs(search_arr - timestamp)
                found_index = np.argmax(search_diff <= dts_err_margin)
                if search_diff[found_index] <= dts_err_margin:
                    if found_index + index < self.size:
                        overflow = True
                        index += found_index
                    else:
                        index = found_index - (self.size - index)
        
        return index, overflow
    
    def segment_current(
        self, 
        offsets: tuple[float, float], 
        error_margin=0.0,
        dts_err_scale=_default_dts_err_scale
    ):
        """Extracts a segment around the current time"""
        return self.segment_index(self.current_index, offsets, error_margin, dts_err_scale)
    
    def segment(
        self, 
        timestamp: float, 
        offsets: tuple[float, float], 
        error_margin=0.0,
        dts_err_scale=_default_dts_err_scale
    ):
        """
        Extracts a segment of the buffer based around a 
        global timestamp and relative offsets.
        """
        
        m_index, _ = self.find_index(timestamp, error_margin, dts_err_scale)
        return self.segment_index(m_index, offsets, error_margin, dts_err_scale)
    
    def segment_index(
        self, 
        m_index: int, 
        offsets: tuple[float, float], 
        error_margin=0.0,
        dts_err_scale=_default_dts_err_scale
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """
        Extracts a segment around relative offsets of the buffer 
        using an index to locate the pivot timestamp. 
        """
        
        if m_index < 0 or m_index > self.size:
            return None, None
        
        x, y = offsets
        buffer_tm = self.timestamps
        tm = buffer_tm[m_index]
        buffer = self.data
        size = self.size
        
        x_index, x_overflow = self.find_index(tm + x, error_margin, dts_err_scale)
        y_index, y_overflow = self.find_index(tm + y, error_margin, dts_err_scale)   
        
        if (
            self.__invalid_indices(m_index, x_index, y_index) or 
            buffer_tm[m_index] < 0 or 
            buffer_tm[x_index] < 0 or 
            buffer_tm[y_index] < 0 or 
            x>y
        ): 
            return None, None
    
        if not (x_overflow or y_overflow) or (x_overflow and y_overflow):
            return buffer[:, x_index:y_index].T, buffer_tm[x_index:y_index]
    
        else:
            start, end = (x_index, y_index) if x_overflow else (y_index, x_index)
            return (
                np.concatenate((buffer[:, start:size], buffer[:, 0:end]), axis=1).T,
                np.concatenate((buffer_tm[start:size], buffer_tm[0:end]))
            )
        
    def __invalid_indices(self, m_index: int, x_index: int, y_index: int):
        """Checks if the indices are out of the buffer"""
        return (
            m_index < 0 or 
            x_index < 0 or 
            y_index < 0 or 
            y_index > self.size or 
            x_index > self.size or 
            m_index > self.size
        )
        
    def __print_segment(self, tm:float, x: float, y: float, start: bool, end: bool):
        buffer_tm = self.timestamps
        print(
            f"""Segment OVERFLOW:
                    Nominal: {self.nominal_srate}
                    Effective: {self.effective_srate}
                    Wanted: [{tm+x}:{tm+y}]
                    Actual: [{buffer_tm[start]}:{buffer_tm[end]}]
                    Error: [{buffer_tm[start]-tm-x}:{buffer_tm[end]-tm-y}]
        """)