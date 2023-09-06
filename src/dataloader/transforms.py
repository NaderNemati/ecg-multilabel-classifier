import numpy as np
import random
from scipy import interpolate
import scipy.io as sio
import scipy.signal as signal
import os
import matplotlib.pyplot as plt
import sys
import copy
import neurokit2 as nk
import math
# Data is expected to be in [channels, samples]
# Notes: some methods apply randomly to channels and some same for all channels
# randomization ranges not carefully checked

class Compose(object):
    def __init__(self, transforms, p = 0.5):
        self.transforms = transforms
        self.all_p = p

    def __call__(self, mseq):
        if self.all_p < np.random.rand(1):
            return mseq
        for t in self.transforms:
            mseq = t(mseq)
        return mseq


class Retype(object):
    def __call__(self, mseq):
        return mseq.astype(np.float32)


class Resample(object):
    def __init__(self, fs_new, fs_old):
        self.fs_new = fs_new
        self.fs_old = fs_old

    def __call__(self, mseq):
        num = int(mseq.shape[1]*self.fs_new/self.fs_old)
        mseq_rs = np.zeros([mseq.shape[0], num])
        for i, row in enumerate(mseq):
            mseq_rs[i,:] = signal.resample(row, num)
        return mseq_rs


class Spline_interpolation(object):
    def __init__(self, fs_new, fs_old):
        self.fs_new = fs_new
        self.fs_old = fs_old

    def spliner(self, ind_orig, val_orig, ind_new):
        spline_fn = interpolate.interp1d(ind_orig, val_orig, kind='cubic')
        return spline_fn(ind_new)

    def __call__(self, mseq):
        n_old = mseq.shape[1]
        n_new = int(n_old*self.fs_new/self.fs_old)

        T = n_old/self.fs_old
        ind_orig = np.linspace(0,T,n_old)
        ind_new = np.linspace(ind_orig[0],ind_orig[-1],n_new)

        mseq_rs = np.zeros([mseq.shape[0], n_new])
        for i, row in enumerate(mseq):
            mseq_rs[i,:] = self.spliner(ind_orig, row, ind_new)
        return mseq_rs


class BandPassFilter(object):
    def __init__(self, fs, lf=0.5, hf=50, order=2):
        self.fs = fs
        self.lf = lf
        self.hf = hf
        self.order = order

    def bpf(self, arr, fs, lf=0.5, hf=50, order=2):
        wbut = [2*lf/fs, 2*hf/fs]
        sos = signal.butter(order, wbut, btype = 'bandpass', output = 'sos')
        return signal.sosfiltfilt(sos, arr, padlen=250, padtype='even')

    def __call__(self, mseq):
        for i, row in enumerate(mseq):
            mseq[i,:] = self.bpf(row, self.fs, self.lf, self.hf, self.order)
        return mseq


class Normalize(object):
    def __init__(self, type="0-1"):
        self.type = type

    def __call__(self, mseq):
        if self.type == "0-1":
            for i, row in enumerate(mseq):
                if sum(mseq[i, :] == 0):
                    mseq[i, :] = mseq[i, :]
                else:
                    mseq[i,:] = (row - np.min(row)) / (np.max(row) - np.min(row))
        elif self.type == "mean-std":
            for i, row in enumerate(mseq):
                mseq[i,:] = (row - np.mean(row)) / np.std(row)
        elif self.type == "none":
            mseq = mseq
        else:
            raise NameError('This normalization is not included!')
        return mseq


class AddNoise(object):

    def __init__(self, sigma=0.3, p = 1):
        self.sigma = sigma
        self.addnoise_p = p

    def __call__(self, mseq):
        if self.addnoise_p < np.random.rand(1):
            return mseq
        sigma = np.random.uniform(0,self.sigma)
        mseq = mseq + np.random.normal(loc=0, scale=sigma, size=mseq.shape)
        return mseq


class Roll(object):

    def __init__(self, n = 250, p=1):
        self.n = n
        self.roll_p = p

    def __call__(self, mseq):
        if self.roll_p < np.random.rand(1):
            return mseq
        sign = np.random.choice([-1,1])
        n = np.random.randint(0, self.n)
        for i, row in enumerate(mseq):
            mseq[i,:] = np.roll(row, sign*n)
        return mseq

class Roll_signal(object):
    def __init__(self, offset=90, p = 1):
        self.offset = offset
        self.roll_p =p

    def __call__(self, mseq):
        if self.roll_p < np.random.rand(1):
            return mseq
        seq = mseq[1,:].ravel()
        _, info = nk.ecg_peaks(seq, method="kalidas2017")
        peaks = info["ECG_R_Peaks"]
        first_peak_location = peaks[0]

        for i, row in enumerate(mseq):
            mseq[i,:] = np.roll(row, first_peak_location)
        return mseq

class Flipy(object):

    def __init__(self, p = 0.5):
        self.flipy_p = p

    def __call__(self, mseq):
        if self.flipy_p < np.random.rand(1):
            return mseq
        for i, row in enumerate(mseq):
            mseq[i,:] = np.multiply(row,-1)
        return mseq


class Flipx(object):

    def __init__(self, p = 0.5):
        self.flipx_p = p

    def __call__(self, mseq):
        if self.flipx_p < np.random.rand(1):
            return mseq
        return np.fliplr(mseq)


class MultiplySine(object):

    def __init__(self, fs = 250, f = 2, a = 1, p = 0.5):
        self.fs = 250
        self.f = f
        self.a = a
        self.multiply_sine_p = p

    def __call__(self, mseq):
        if self.multiply_sine_p < np.random.rand(1):
            return mseq
        t = np.arange(mseq.shape[1])/self.fs
        for i, row in enumerate(mseq):
            f, a = np.random.uniform(0,self.f), np.random.uniform(0,self.a)
            mseq[i,:] = row*(1 + a*np.sin(2*np.pi*f*t))
        return mseq


class MultiplyLinear(object):

    def __init__ (self, multiplier = 2.5, p = 1):
        self.multiply_linear_p = p
        self.multiplier = multiplier

    def __call__(self, mseq):
        if self.multiply_linear_p < np.random.rand(1):
            return mseq
        n = mseq.shape[1]
        for i, row in enumerate(mseq):
            #m = np.random.uniform(1,self.multiplier,2)
            #v = np.linspace(m[0],m[1],n)
            v = np.linspace(1, self.multiplier, n)
            mseq[i,:] = np.multiply(row, v)
        return mseq


class MultiplyTriangle(object):

    def __init__(self, scale = 4, p = 1):
        self.multiply_triangle_p = p
        self.scale = scale

    def __call__(self, mseq):
        if self.multiply_triangle_p < np.random.rand(1):
            return mseq
        n_samples = mseq.shape[1]
        for i, row in enumerate(mseq):
            n_turning_point = int(np.random.uniform(0,1)*n_samples)
            m = np.random.uniform(1/self.scale, self.scale)
            v1 = np.linspace(1,m,n_turning_point)
            v2 = np.linspace(m,1,n_samples - n_turning_point)
            v = np.concatenate([v1,v2])
            mseq[i,:] = np.multiply(row, v)
        return mseq


class RandomClip(object):
    def __init__(self, w=1000):
        self.w = w

    def __call__(self, mseq):
        if mseq.shape[1] >= self.w:
            start = random.randint(0, mseq.shape[1] - self.w)
            mseq = mseq[:, start:start + self.w]
        else:
            left = random.randint(0, self.w - mseq.shape[1])
            right = self.w - mseq.shape[1] - left
            zeros_padding1 = np.zeros(shape=(mseq.shape[0], left))
            zeros_padding2 = np.zeros(shape=(mseq.shape[0], right))
            mseq = np.hstack((zeros_padding1, mseq, zeros_padding2))
        return mseq


class RandomStretch(object):
    def __init__(self, scale=1.5, p = 0.5):
        self.scale = scale
        self.p = p

    def __call__(self, mseq):
        if self.p < np.random.rand(1):
            return mseq
        m = np.random.uniform(1/self.scale, self.scale)
        num = int(mseq.shape[1]*m)
        for i, row in enumerate(mseq):
            y = signal.resample(row, num)
            if len(y) < len(row):
                mseq[i,:len(y)] = y
            else:
                mseq[i,:] = y[:len(row)]
            return mseq


class ResampleSine(object):
    def __init__(self, fs = 250, freq_lo = 0.0, freq_hi = 0.3,
                 scale_lo = 0.0, scale_hi = 0.5, p=0.5):
        self.fs = fs
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi
        self.p = p

    def __call__(self, mseq):
        if self.p < np.random.rand(1):
            return mseq
        scale = np.random.uniform(self.scale_lo, self.scale_hi)
        freq = np.random.uniform(self.freq_lo, self.freq_hi)
        x_orig = np.arange(0, mseq.shape[1])/self.fs
        x_new = x_orig + scale*np.sin(2*np.pi*freq*x_orig)
        for i, row in enumerate(mseq):
            mseq[i,:] = np.interp(x_new, x_orig, row)
        return mseq


class ResampleLinear(object):
    def __init__(self, scale = 2, p=0.5):
        self.scale = scale
        self.p = p

    def __call__(self, mseq):
        if self.p < np.random.rand(1):
            return mseq
        x_orig = np.arange(0, mseq.shape[1])
        scale = np.random.uniform(self.scale, self.scale)
        scale = np.linspace(1,scale,len(x_orig))
        x_new = x_orig*scale
        x_new = x_new*(x_orig[-1]/x_new[-1])
        for i, row in enumerate(mseq):
            mseq[i,:] = np.interp(x_new, x_orig, row)
        return mseq


class NotchFilter(object):
    def __init__(self, fs, Q = 1, p = 0.5):
        self.fs = fs
        self.Q = Q
        self.p = p

    def nf(self, arr, fs, f0, Q):
        b, a = signal.iirnotch(f0, Q, fs)
        return signal.filtfilt(b, a, arr)

    def __call__(self, mseq):
        if self.p < np.random.rand(1):
            return mseq
        #f0 = np.random.uniform(1, int(self.fs/2))
        f0 = 20
        for i, row in enumerate(mseq):
            mseq[i,:] = self.nf(row, self.fs, f0, self.Q)
        return mseq


class ValClip(object):
    def __init__(self, w=72000):
        self.w = w

    def __call__(self, seq):
        if seq.shape[1] >= self.w:
            seq = seq
        else:
            zeros_padding = np.zeros(shape=(seq.shape[0], self.w - seq.shape[1]), dtype=np.float32)
            seq = np.hstack((seq, zeros_padding))
        return seq



import numpy as np
from scipy import interpolate

class ResampleLinearAlign1stPeak(object):
    def __init__(self, scale=1.2, p=1, w=4096):
        self.scale = scale
        self.p = p
        self.w = w

    def __call__(self, signal):
        signal = np.copy(signal)
        if self.p < np.random.rand(1):
            return signal
        original_length = signal.shape[1]
        new_length = int(original_length * self.scale)
        x_orig = np.linspace(0, original_length - 1, original_length)

        # Generate a non-linearly spaced array for x
        x_new = np.linspace(0, 1, new_length)
        # Updated line
        x_new = np.power(x_new, self.scale)
        x_new *= (original_length - 1)

        signal = np.flip(signal)

        # Initialize signal_resampled and assign shape
        signal_resampled = np.zeros((signal.shape[0], new_length))

        for i, row in enumerate(signal):
            interpolator = interpolate.interp1d(x_orig, row)
            signal_resampled[i, :] = interpolator(x_new)

        # Reverse the order of the resampled signal
        signal_resampled = np.flip(signal_resampled)

        # Truncate or pad the signal_resampled array
        if signal_resampled.shape[1] > self.w:
            signal_resampled = signal_resampled[:, :self.w]
        elif signal_resampled.shape[1] < self.w:
            padded_signal = np.zeros((signal_resampled.shape[0], self.w))
            padded_signal[:, :signal_resampled.shape[1]] = signal_resampled
            signal_resampled = padded_signal

        return signal_resampled







class ResampleTriangle(object):
    def __init__(self, scale=1.35, p=1):
        self.p = p
        self.scale = scale
        self.scale_range = np.linspace(1 / self.scale, self.scale, num=1000)
        self.scale1 = np.random.choice(self.scale_range)
        self.scale2 = math.log(2.5) / self.scale1
        self.resample_linear1 = ResampleLinear(self.scale1, self.p)
        self.resample_linear2 = ResampleLinear(self.scale2, self.p)

    def __call__(self, mseq):
        mseq = copy.deepcopy(mseq)
        if self.p < np.random.rand(1):
            return mseq

        for i, row in enumerate(mseq):
            mid = len(row) // 2
            mseq_m1 = row[:mid]
            mseq_m2 = row[mid + 1:]

            mseq_m1_rti = self.resample_linear1(np.array([mseq_m1]))
            mseq_m2_rti = self.resample_linear2(np.array([mseq_m2]))

            new_row = np.empty(mseq_m1_rti.shape[1] + mseq_m2_rti.shape[1])
            new_row[:mseq_m1_rti.shape[1]] = mseq_m1_rti
            new_row[mseq_m1_rti.shape[1]:] = mseq_m2_rti
            mseq[i, :] = np.resize(new_row, (mseq.shape[1],))

        return mseq


class EqualSegmentResampler(object):
    def __init__(self, A2=1.3, A3=0.8, p=1):
        self.A2 = A2
        self.A3 = A3
        self.p = p


    def Peak_detection(self, mseq):
        for i in range(mseq.shape[0]):
            cleaned = nk.ecg_clean(mseq[i, :], method="kalidas2017")
            _, info = nk.ecg_peaks(cleaned, method="kalidas2017")
            Peaks_part = info["ECG_R_Peaks"]
            if np.any(np.isnan(Peaks_part)) or len(Peaks_part) == 0:
                continue
            else:
                break
   
        firstPart = []  # Initialize firstPart as an empty list
        lastPart = []  # Initialize lastPart as an empty list
   
        for row in mseq:
            first = row[0:Peaks_part[0]]
            last = row[Peaks_part[-1]:]
            firstPart.append(first)
            lastPart.append(last)
   
        return Peaks_part, firstPart, lastPart    

    def separate_segments(self, signal_1st, Peaks_part):
        segments = []
        for i in range(len(Peaks_part) - 1):
            start_index = Peaks_part[i]
            end_index = Peaks_part[i+1]
            segment = signal_1st[start_index:end_index]
            segments.append(segment)
        return segments    


    def Resamling_Coefficients(self, A2, A3, Peaks):
        A1 = Peaks.shape[0]-1
        x = np.arange(5, 7, (7-5)/A1)
        y = np.sin(x)
        sumation = np.sum(y)
        minimmum = np.min(y)
        y1 = y + (-1 * minimmum)
        maaximmum = np.max(y1)
        y2 = y1 * ((A2 - A3) / maaximmum)
        y3 = y2 + A3
        coeff = np.array((y3 * (len(y3) / np.sum(y3))))
        return coeff

    def __call__(self, mseq):

        resampled = np.zeros_like(mseq)  # Initialize resample_signal array
            # Peak detection
            # Split the parts of signal before the Peak and after the last Peak
        Peaks_part, firstPart, lastPart = self.Peak_detection(mseq)
        for i, row in enumerate(mseq):
            mseq[i, :] = copy.deepcopy(row)
            if self.p < np.random.rand(1):
                return mseq

            segments = self.separate_segments(row, Peaks_part)

            seg_sum = 0
            for segment in segments:
                seg_sum = seg_sum + len(segment)
                #print("Sumation of all segments length:", seg_sum)
                # Calculating the desired lengh of each segment
            B1 = seg_sum / ((Peaks_part.shape[0]) - 1)
            S1 = []
            S2 = []
            S3 = 0
            for B2 in np.arange(0, seg_sum, B1):
                S3 = S3 + 1
                S1.append(np.round(B2))
                if S3 > 1:
                    S2.append(S1[S3 - 1] - S1[S3 - 2])
            S2.append(seg_sum - np.sum(S2))
                #print("len of S2", len(S2))
                #print("sum of S2", np.sum(S2))
                #print("S2", S2)
                #print("B1: ", B1)
                # Resampling all segments to achieve desired lengths that must be approximately equal
            eq_s = []
            for j, s in zip(range(len(segments)), S2):
                b = nk.signal_resample(segments[j], desired_length=s, sampling_rate=None, desired_sampling_rate=None,
                                           method='interpolation')
                eq_s.append(b)
               
            coeff = self.Resamling_Coefficients(A2=1.3, A3=0.8, Peaks=Peaks_part)
       
                # Resampling segments with desired coefficients
            res_seg = []
            for t, c in zip(range(len(eq_s)), coeff):
                a = nk.signal_resample(eq_s[t], desired_length=int(len(eq_s[t]) * c), sampling_rate=None,
                                           desired_sampling_rate=None, method='interpolation')
                res_seg.append(a)
                   
                #Applying the same coefficients on the heights of segments    
            diff_height = []    
            for s, c in zip(res_seg, coeff):
                f = s*c
                diff_height.append(f)
               
                #print('diff_height:', diff_height)
                #print('len of diff_height:', diff_height)
                #print('type of diff_height:', type(diff_height))
               
            ss = 0
            for f in diff_height:
                ss = ss + len(f)
                #print("Sum of res_seg", ss)
            diff = np.sum(S2) - ss
                #print("diff", diff)
            resample_sig = np.array([])
            if diff > 0:
                l = nk.signal_resample(lastPart[i], desired_length=(diff + len(lastPart[i])), sampling_rate=None,
                                           desired_sampling_rate=None, method='interpolation')
                #print('l:', l)
                   
                # Concatenation of resampled segments to achieve resampled signal
                resample_sig = np.concatenate([firstPart[i], np.concatenate(diff_height), l])
            elif diff == 0:
                resample_sig = np.concatenate([firstPart[i], np.concatenate(diff_height)])
            resampled[i,:] = resample_sig
        return resampled
