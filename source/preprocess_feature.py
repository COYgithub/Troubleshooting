# -*- coding: utf-8 -*-
"""
高速列车轴承数据预处理与特征提取脚本

功能：
1. 读取.mat文件和txt/csv文件
2. 数据重采样到48kHz
3. 滑动窗口分段处理
4. 提取时域、频域、包络谱和时频特征
5. 保存特征到CSV文件
"""
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import io, signal
from scipy.fft import fft, fftfreq
from scipy.signal import hilbert, stft, resample
import pywt
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

code_dir = os.getcwd()
root_dir = os.path.dirname(code_dir)
data_dir = os.path.join(root_dir, '数据集')

class BearingDataProcessor:
    """轴承数据处理类"""

    def __init__(self, target_fs=48000, window_size=1.0, step_size=0.5):
        """
        初始化参数

        Args:
            target_fs: 目标采样率 (Hz)
            window_size: 窗口大小 (秒)
            step_size: 步长 (秒)
        """
        self.target_fs = target_fs
        self.window_size = window_size
        self.step_size = step_size
        self.window_samples = int(window_size * target_fs)
        self.step_samples = int(step_size * target_fs)

        # SKF轴承参数（根据数据集规格）
        self.bearing_params = {
            'SKF6205': {  # 驱动端轴承
                'pitch_diameter': 1.537 * 25.4,  # 1.537英寸转换为mm
                'ball_diameter': 0.3126 * 25.4,  # 0.3126英寸转换为mm
                'num_balls': 9,
                'contact_angle': 0
            },
            'SKF6203': {  # 风扇端轴承
                'pitch_diameter': 1.122 * 25.4,  # 1.122英寸转换为mm
                'ball_diameter': 0.2656 * 25.4,  # 0.2656英寸转换为mm
                'num_balls': 9,
                'contact_angle': 0
            }
        }

    def calculate_bearing_frequencies(self, rpm, bearing_type='SKF6205'):
        """
        计算轴承特征频率

        Args:
            rpm: 转速 (转/分)
            bearing_type: 轴承型号 ('SKF6205' 或 'SKF6203')

        Returns:
            dict: 包含BPFO, BPFI, BSF频率的字典
        """
        fr = rpm / 60  # 轴频率 (Hz)

        # 获取轴承几何参数
        params = self.bearing_params.get(bearing_type, self.bearing_params['SKF6205'])
        d = params['ball_diameter']
        D = params['pitch_diameter']
        n = params['num_balls']
        alpha = np.radians(params['contact_angle'])

        # 特征频率计算
        bpfo = (n * fr / 2) * (1 - (d / D) * np.cos(alpha))  # 外圈故障频率
        bpfi = (n * fr / 2) * (1 + (d / D) * np.cos(alpha))  # 内圈故障频率
        bsf = (D * fr / (2 * d)) * (1 - ((d / D) * np.cos(alpha)) ** 2)  # 滚动体故障频率

        return {'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf, 'FR': fr}

    def read_mat_file(self, file_path):
        """
        读取.mat文件（适配您的数据集格式）

        Args:
            file_path: 文件路径

        Returns:
            dict: 包含信号数据和RPM信息
        """
        try:
            mat_data = io.loadmat(file_path)

            data = {}

            # 查找所有非内置键
            data_keys = [key for key in mat_data.keys() if not key.startswith('__')]

            # 解析数据键
            for key in data_keys:
                key_lower = key.lower()
                if 'de' in key_lower and 'time' in key_lower:
                    data['DE'] = mat_data[key].flatten()
                elif 'fe' in key_lower and 'time' in key_lower:
                    data['FE'] = mat_data[key].flatten()
                elif 'ba' in key_lower and 'time' in key_lower:
                    data['BA'] = mat_data[key].flatten()
                elif 'rpm' in key_lower:
                    rpm_val = mat_data[key]
                    if hasattr(rpm_val, 'flatten'):
                        data['RPM'] = float(rpm_val.flatten()[0])
                    else:
                        data['RPM'] = float(rpm_val)

            # 从文件名推断故障类型和尺寸
            filename = os.path.basename(file_path)
            data['fault_info'] = self._parse_filename(filename)

            # 根据数据长度推断采样率
            if 'DE' in data:
                data_length = len(data['DE'])
                # 48kHz数据通常长度更长
                if data_length > 100000:
                    data['fs'] = 48000
                else:
                    data['fs'] = 12000
            elif 'FE' in data:
                data_length = len(data['FE'])
                data['fs'] = 12000  # FE数据通常是12kHz
            else:
                data['fs'] = 48000  # 默认

            print(f"读取文件: {filename}")
            print(f"  数据键: {list(data.keys())}")
            if 'DE' in data:
                print(f"  DE数据长度: {len(data['DE'])}")
            if 'FE' in data:
                print(f"  FE数据长度: {len(data['FE'])}")
            if 'BA' in data:
                print(f"  BA数据长度: {len(data['BA'])}")
            print(f"  采样率: {data['fs']} Hz")
            if 'RPM' in data:
                print(f"  转速: {data['RPM']} RPM")

            return data

        except Exception as e:
            print(f"读取.mat文件出错 {file_path}: {e}")
            return None

    def _parse_filename(self, filename):
        """
        解析文件名获取故障信息

        Args:
            filename: 文件名

        Returns:
            dict: 故障信息字典
        """
        info = {
            'fault_type': 'N',  # 默认正常
            'fault_size': 0,
            'load': 0,
            'position': None
        }

        # 移除扩展名
        name = filename.replace('.mat', '').upper()

        # 解析故障类型
        if name.startswith('B'):
            info['fault_type'] = 'B'  # 滚动体故障
        elif name.startswith('IR') or name.startswith('R'):
            info['fault_type'] = 'IR'  # 内圈故障
        elif name.startswith('OR'):
            info['fault_type'] = 'OR'  # 外圈故障
        elif name.startswith('N'):
            info['fault_type'] = 'N'  # 正常状态

        # 解析故障尺寸
        if '007' in name:
            info['fault_size'] = 0.007
        elif '014' in name:
            info['fault_size'] = 0.014
        elif '021' in name:
            info['fault_size'] = 0.021
        elif '028' in name:
            info['fault_size'] = 0.028

        # 解析载荷
        parts = name.split('_')
        for part in parts:
            if part.isdigit():
                info['load'] = int(part)
                break

        # 解析外圈故障位置
        if info['fault_type'] == 'OR':
            if 'CENTERED' in name or 'CENTRED' in name:
                info['position'] = 'Centered'
            elif 'ORTHOGONAL' in name:
                info['position'] = 'Orthogonal'
            elif 'OPPOSITE' in name:
                info['position'] = 'Opposite'

        return info

    def read_csv_txt_file(self, file_path):
        """
        读取CSV或TXT文件

        Args:
            file_path: 文件路径

        Returns:
            dict: 包含信号数据信息
        """
        try:
            if file_path.endswith('.csv'):
                data_df = pd.read_csv(file_path)
                signal_data = data_df.iloc[:, 0].values  # 假设第一列是信号数据
            else:
                signal_data = np.loadtxt(file_path)
                if signal_data.ndim > 1:
                    signal_data = signal_data[:, 0]  # 取第一列

            return {
                'signal': signal_data,
                'fs': 32000,  # 列车数据采样率
                'duration': len(signal_data) / 32000
            }

        except Exception as e:
            print(f"读取文件出错: {e}")
            return None

    def resample_signal(self, signal_data, original_fs):
        """
        重采样信号到目标采样率

        Args:
            signal_data: 原始信号
            original_fs: 原始采样率

        Returns:
            重采样后的信号
        """
        if original_fs != self.target_fs:
            num_samples = int(len(signal_data) * self.target_fs / original_fs)
            resampled_signal = resample(signal_data, num_samples)
            return resampled_signal
        return signal_data

    def sliding_window_segmentation(self, signal_data):
        """
        滑动窗口分段

        Args:
            signal_data: 输入信号

        Returns:
            list: 分段后的信号列表
        """
        segments = []
        start_idx = 0

        while start_idx + self.window_samples <= len(signal_data):
            segment = signal_data[start_idx:start_idx + self.window_samples]
            segments.append(segment)
            start_idx += self.step_samples

        return segments

    def extract_time_features(self, segment):
        """
        提取时域特征

        Args:
            segment: 信号段

        Returns:
            dict: 时域特征字典
        """
        features = {}

        # 基础统计特征
        features['mean'] = np.mean(segment)
        features['std'] = np.std(segment)
        features['rms'] = np.sqrt(np.mean(segment ** 2))
        features['peak'] = np.max(np.abs(segment))
        features['skewness'] = self._skewness(segment)
        features['kurtosis'] = self._kurtosis(segment)

        # 形状指标
        features['crest_factor'] = features['peak'] / features['rms']
        features['impulse_factor'] = features['peak'] / np.mean(np.abs(segment))
        features['shape_factor'] = features['rms'] / np.mean(np.abs(segment))
        features['clearance_factor'] = features['peak'] / (np.mean(np.sqrt(np.abs(segment)))) ** 2

        return features

    def _skewness(self, x):
        """计算偏度"""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        return np.sum(((x - mean) / std) ** 3) / n if std > 0 else 0

    def _kurtosis(self, x):
        """计算峭度"""
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        return np.sum(((x - mean) / std) ** 4) / n - 3 if std > 0 else 0

    def extract_frequency_features(self, segment, bearing_freqs=None):
        """
        提取频域特征

        Args:
            segment: 信号段
            bearing_freqs: 轴承特征频率字典

        Returns:
            dict: 频域特征字典
        """
        features = {}

        # FFT计算
        fft_values = fft(segment)
        fft_magnitude = np.abs(fft_values[:len(fft_values) // 2])
        freqs = fftfreq(len(segment), 1 / self.target_fs)[:len(fft_values) // 2]

        # 能量谱特征
        power_spectrum = fft_magnitude ** 2
        features['total_energy'] = np.sum(power_spectrum)

        # 谱质心和谱带宽
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)
        centroid = features['spectral_centroid']
        features['spectral_bandwidth'] = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * fft_magnitude) / np.sum(fft_magnitude))

        # 频带能量分布
        freq_bands = [(0, 1000), (1000, 5000), (5000, 10000), (10000, 24000)]
        for i, (low, high) in enumerate(freq_bands):
            band_mask = (freqs >= low) & (freqs <= high)
            features[f'energy_band_{i + 1}'] = np.sum(power_spectrum[band_mask])

        # 轴承特征频率处的幅值
        if bearing_freqs:
            for freq_name, freq_val in bearing_freqs.items():
                if freq_val < self.target_fs / 2:  # 低于奈奎斯特频率
                    # 找最近频率点
                    freq_idx = np.argmin(np.abs(freqs - freq_val))
                    features[f'{freq_name}_amplitude'] = fft_magnitude[freq_idx]

                    # 在特征频率附近的能量
                    freq_range = 10  # Hz
                    freq_mask = (freqs >= freq_val - freq_range) & (freqs <= freq_val + freq_range)
                    features[f'{freq_name}_energy'] = np.sum(power_spectrum[freq_mask])

        return features

    def extract_envelope_features(self, segment):
        """
        提取包络谱特征

        Args:
            segment: 信号段

        Returns:
            dict: 包络特征字典
        """
        features = {}

        # Hilbert变换获取包络
        analytic_signal = hilbert(segment)
        envelope = np.abs(analytic_signal)

        # 包络统计特征
        features['envelope_mean'] = np.mean(envelope)
        features['envelope_std'] = np.std(envelope)
        features['envelope_rms'] = np.sqrt(np.mean(envelope ** 2))
        features['envelope_peak'] = np.max(envelope)

        # 包络谱
        envelope_fft = fft(envelope - np.mean(envelope))
        envelope_magnitude = np.abs(envelope_fft[:len(envelope_fft) // 2])
        envelope_freqs = fftfreq(len(envelope), 1 / self.target_fs)[:len(envelope_fft) // 2]

        # 包络谱能量
        features['envelope_spectral_energy'] = np.sum(envelope_magnitude ** 2)

        # 包络谱质心
        if np.sum(envelope_magnitude) > 0:
            features['envelope_spectral_centroid'] = np.sum(envelope_freqs * envelope_magnitude) / np.sum(
                envelope_magnitude)
        else:
            features['envelope_spectral_centroid'] = 0

        return features

    def extract_timefreq_features(self, segment, save_images=False, filename_prefix=""):
        """
        提取时频特征

        Args:
            segment: 信号段
            save_images: 是否保存时频图
            filename_prefix: 文件名前缀

        Returns:
            dict: 时频特征字典
        """
        features = {}

        # STFT
        f_stft, t_stft, Zxx = stft(segment, fs=self.target_fs, nperseg=1024)
        stft_magnitude = np.abs(Zxx)

        # STFT特征
        features['stft_energy'] = np.sum(stft_magnitude ** 2)
        features['stft_peak_freq'] = f_stft[np.unravel_index(np.argmax(stft_magnitude), stft_magnitude.shape)[0]]

        # 时频集中度
        total_energy = np.sum(stft_magnitude ** 2)
        if total_energy > 0:
            time_spread = np.sum(np.var(stft_magnitude, axis=0)) / total_energy
            freq_spread = np.sum(np.var(stft_magnitude, axis=1)) / total_energy
            features['time_concentration'] = 1 / (1 + time_spread)
            features['freq_concentration'] = 1 / (1 + freq_spread)
        else:
            features['time_concentration'] = 0
            features['freq_concentration'] = 0

        # 连续小波变换 (CWT)
        scales = np.arange(1, 128)
        wavelet = 'cmor'
        coefficients, frequencies = pywt.cwt(segment, scales, wavelet, sampling_period=1 / self.target_fs)
        cwt_magnitude = np.abs(coefficients)

        # CWT特征
        features['cwt_energy'] = np.sum(cwt_magnitude ** 2)
        features['cwt_peak_freq'] = frequencies[np.unravel_index(np.argmax(cwt_magnitude), cwt_magnitude.shape)[0]]

        # 保存时频图
        if save_images and filename_prefix:
            self._save_timefreq_plots(segment, f_stft, t_stft, stft_magnitude,
                                      frequencies, cwt_magnitude, filename_prefix)

        return features

    def _save_timefreq_plots(self, segment, f_stft, t_stft, stft_mag, cwt_freqs, cwt_mag, prefix):
        """保存时频分析图像"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 原始信号
        axes[0, 0].plot(segment)
        axes[0, 0].set_title('原始信号')
        axes[0, 0].set_xlabel('样本点')
        axes[0, 0].set_ylabel('幅值')

        # STFT
        axes[0, 1].pcolormesh(t_stft, f_stft, 20 * np.log10(stft_mag + 1e-10))
        axes[0, 1].set_title('STFT')
        axes[0, 1].set_xlabel('时间 (s)')
        axes[0, 1].set_ylabel('频率 (Hz)')

        # CWT
        axes[1, 0].pcolormesh(np.arange(len(segment)) / self.target_fs, cwt_freqs,
                              20 * np.log10(cwt_mag + 1e-10))
        axes[1, 0].set_title('CWT')
        axes[1, 0].set_xlabel('时间 (s)')
        axes[1, 0].set_ylabel('频率 (Hz)')

        # 频谱
        fft_vals = fft(segment)
        freqs = fftfreq(len(segment), 1 / self.target_fs)
        axes[1, 1].semilogy(freqs[:len(freqs) // 2], np.abs(fft_vals[:len(fft_vals) // 2]))
        axes[1, 1].set_title('频谱')
        axes[1, 1].set_xlabel('频率 (Hz)')
        axes[1, 1].set_ylabel('幅值')

        plt.tight_layout()
        plt.savefig(f'{prefix}_timefreq.png', dpi=150, bbox_inches='tight')
        plt.close()

    def process_file(self, file_path, label=None, rpm=None, save_images=False):
        """
        处理单个文件

        Args:
            file_path: 文件路径
            label: 手动指定标签 (N, OR, IR, B)，如果为None则从文件名自动解析
            rpm: 手动指定转速，如果为None则从文件中读取
            save_images: 是否保存时频图

        Returns:
            pd.DataFrame: 特征数据框
        """
        print(f"\n处理文件: {file_path}")

        # 读取文件
        if file_path.endswith('.mat'):
            data = self.read_mat_file(file_path)
            if data is None:
                return None

            # 获取标签信息
            if label is None:
                label = data['fault_info']['fault_type']

            # 获取转速信息
            if rpm is None:
                rpm = data.get('RPM', 1797)  # 默认转速

            # 处理多传感器数据
            all_features = []
            sensors = ['DE', 'FE', 'BA']

            for sensor in sensors:
                if sensor in data:
                    signal_data = data[sensor]
                    original_fs = data.get('fs', 12000)

                    print(f"  处理{sensor}传感器数据，原始采样率: {original_fs} Hz")

                    # 重采样
                    resampled_signal = self.resample_signal(signal_data, original_fs)
                    print(f"  重采样后长度: {len(resampled_signal)} 点")

                    # 分段处理
                    segments = self.sliding_window_segmentation(resampled_signal)
                    print(f"  分段数量: {len(segments)}")

                    # 确定轴承类型
                    bearing_type = 'SKF6205' if sensor == 'DE' else 'SKF6203'

                    # 计算轴承特征频率
                    bearing_freqs = self.calculate_bearing_frequencies(rpm, bearing_type)

                    # 提取特征
                    features_list = []
                    for i, segment in enumerate(segments):
                        features = {}
                        features['file'] = os.path.basename(file_path)
                        features['sensor'] = sensor
                        features['segment'] = i
                        features['label'] = label
                        features['fault_size'] = data['fault_info']['fault_size']
                        features['load'] = data['fault_info']['load']
                        features['position'] = data['fault_info']['position']
                        features['rpm'] = rpm
                        features['bearing_type'] = bearing_type

                        # 各类特征
                        time_features = self.extract_time_features(segment)
                        freq_features = self.extract_frequency_features(segment, bearing_freqs)
                        envelope_features = self.extract_envelope_features(segment)

                        filename_prefix = f"{os.path.splitext(os.path.basename(file_path))[0]}_{sensor}_seg{i}"
                        timefreq_features = self.extract_timefreq_features(segment, save_images, filename_prefix)

                        # 合并特征
                        features.update(time_features)
                        features.update(freq_features)
                        features.update(envelope_features)
                        features.update(timefreq_features)

                        features_list.append(features)

                    all_features.extend(features_list)

            return pd.DataFrame(all_features)

        else:
            # 处理CSV/TXT文件 (目标域数据)
            data = self.read_csv_txt_file(file_path)
            if data is None:
                return None

            signal_data = data['signal']
            original_fs = data['fs']

            # 从文件名推断标签 (A-P编号)
            filename = os.path.basename(file_path)
            file_id = filename.split('.')[0].split('_')[0]

            print(f"  目标域数据，文件ID: {file_id}")
            print(f"  原始采样率: {original_fs} Hz，数据长度: {len(signal_data)} 点")

            # 重采样
            resampled_signal = self.resample_signal(signal_data, original_fs)
            print(f"  重采样后长度: {len(resampled_signal)} 点")

            # 分段处理
            segments = self.sliding_window_segmentation(resampled_signal)
            print(f"  分段数量: {len(segments)}")

            # 计算轴承特征频率 (使用列车轴承转速约600 rpm)
            target_rpm = rpm if rpm is not None else 600
            bearing_freqs = self.calculate_bearing_frequencies(target_rpm, 'SKF6205')

            # 提取特征
            features_list = []
            for i, segment in enumerate(segments):
                features = {}
                features['file'] = os.path.basename(file_path)
                features['file_id'] = file_id
                features['sensor'] = 'single'
                features['segment'] = i
                features['label'] = label if label else 'unknown'
                features['fault_size'] = 0
                features['load'] = 0
                features['position'] = None
                features['rpm'] = target_rpm
                features['bearing_type'] = 'target_domain'

                # 各类特征
                time_features = self.extract_time_features(segment)
                freq_features = self.extract_frequency_features(segment, bearing_freqs)
                envelope_features = self.extract_envelope_features(segment)

                filename_prefix = f"{file_id}_seg{i}"
                timefreq_features = self.extract_timefreq_features(segment, save_images, filename_prefix)

                # 合并特征
                features.update(time_features)
                features.update(freq_features)
                features.update(envelope_features)
                features.update(timefreq_features)

                features_list.append(features)

            return pd.DataFrame(features_list)

    def process_dataset(self, data_config, output_dir='./features', save_images=False):
        """
        处理整个数据集

        Args:
            data_config: 数据配置字典
            output_dir: 输出目录
            save_images: 是否保存时频图

        Returns:
            pd.DataFrame: 完整特征数据框
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_features = []

        for config in data_config:
            file_path = config['path']
            label = config.get('label', 'unknown')
            rpm = config.get('rpm', None)

            features_df = self.process_file(file_path, label, rpm, save_images)
            if features_df is not None:
                all_features.append(features_df)

        # 合并所有特征
        if all_features:
            final_df = pd.concat(all_features, ignore_index=True)

            # 保存特征
            output_file = os.path.join(output_dir, 'extracted_features.csv')
            final_df.to_csv(output_file, index=False)
            print(f"特征已保存到: {output_file}")
            print(f"总特征数: {len(final_df.columns)}")
            print(f"总样本数: {len(final_df)}")

            # 分别保存源域和目标域特征
            if 'file_id' in final_df.columns:
                # 分离源域和目标域数据
                source_df = final_df[final_df['file_id'].isna()]
                target_df = final_df[~final_df['file_id'].isna()]

                if len(source_df) > 0:
                    source_file = os.path.join(output_dir, 'source_domain_features.csv')
                    source_df.to_csv(source_file, index=False)
                    print(f"源域特征已保存: {len(source_df)} 个样本")

                if len(target_df) > 0:
                    target_file = os.path.join(output_dir, 'target_domain_features.csv')
                    target_df.to_csv(target_file, index=False)
                    print(f"目标域特征已保存: {len(target_df)} 个样本")
        else:
            print("没有成功处理任何文件")
            return None

        # 特征标准化（可选）
        print("\n进行特征标准化...")
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        # 排除元数据列
        exclude_cols = ['segment', 'fault_size', 'load', 'rpm']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]

        if len(numeric_features) > 0:
            scaler = StandardScaler()
            features_df_normalized = features_df.copy()
            features_df_normalized[numeric_features] = scaler.fit_transform(features_df[numeric_features])

            # 保存标准化后的特征
            features_df_normalized.to_csv('./extracted_features/normalized_features.csv', index=False)
            print(f"标准化特征已保存: {len(numeric_features)} 个数值特征")

            # 保存标准化器
            joblib.dump(scaler, './extracted_features/feature_scaler.pkl')
            print("特征标准化器已保存")

        # 生成特征报告
        generate_feature_report(features_df, './extracted_features/feature_report.txt')

        return final_df


def generate_feature_report(df, output_path):
    """生成特征提取报告"""
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write("=== 轴承振动数据特征提取报告 ===\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")

        # 基本统计
        f.write("1. 数据统计\n")
        f.write(f"   总样本数: {len(df)}\n")
        f.write(f"   总特征数: {len(df.columns)}\n\n")

        # 标签分布
        f.write("2. 标签分布\n")
        label_counts = df['label'].value_counts()
        for label, count in label_counts.items():
            f.write(f"   {label}: {count} 个样本\n")
        f.write("\n")

        # 传感器分布
        if 'sensor' in df.columns:
            f.write("3. 传感器分布\n")
            sensor_counts = df['sensor'].value_counts()
            for sensor, count in sensor_counts.items():
                f.write(f"   {sensor}: {count} 个样本\n")
            f.write("\n")

        # 轴承类型分布
        if 'bearing_type' in df.columns:
            f.write("4. 轴承类型分布\n")
            bearing_counts = df['bearing_type'].value_counts()
            for bearing, count in bearing_counts.items():
                f.write(f"   {bearing}: {count} 个样本\n")
            f.write("\n")

        # 特征类别统计
        f.write("5. 特征类别统计\n")
        time_features = [col for col in df.columns if any(x in col.lower() for x in
                                                          ['mean', 'std', 'rms', 'peak', 'skew', 'kurt', 'crest',
                                                           'impulse', 'shape', 'clear'])]
        freq_features = [col for col in df.columns if
                         any(x in col.lower() for x in ['energy', 'spectral', 'amplitude', 'bpfo', 'bpfi', 'bsf'])]
        envelope_features = [col for col in df.columns if 'envelope' in col.lower()]
        timefreq_features = [col for col in df.columns if
                             any(x in col.lower() for x in ['stft', 'cwt', 'concentration'])]

        f.write(f"   时域特征: {len(time_features)} 个\n")
        f.write(f"   频域特征: {len(freq_features)} 个\n")
        f.write(f"   包络特征: {len(envelope_features)} 个\n")
        f.write(f"   时频特征: {len(timefreq_features)} 个\n\n")

        # 数值特征统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['segment', 'fault_size', 'load', 'rpm']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(numeric_cols) > 0:
            f.write("6. 数值特征统计摘要\n")
            stats = df[numeric_cols].describe()
            f.write(f"   均值范围: [{stats.loc['mean'].min():.6f}, {stats.loc['mean'].max():.6f}]\n")
            f.write(f"   标准差范围: [{stats.loc['std'].min():.6f}, {stats.loc['std'].max():.6f}]\n")
            f.write(f"   最小值范围: [{stats.loc['min'].min():.6f}, {stats.loc['min'].max():.6f}]\n")
            f.write(f"   最大值范围: [{stats.loc['max'].min():.6f}, {stats.loc['max'].max():.6f}]\n\n")

        f.write("7. 处理配置\n")
        f.write("   目标采样率: 48000 Hz\n")
        f.write("   窗口大小: 1.0 秒\n")
        f.write("   步长: 0.5 秒\n")
        f.write("   轴承参数: SKF6205(DE), SKF6203(FE)\n\n")

        f.write("报告生成完毕。\n")

    print(f"特征报告已保存到: {output_path}")

def main():
    """主函数示例 - 适配您的数据集结构"""
    # 初始化处理器
    processor = BearingDataProcessor(
        target_fs=48000,
        window_size=1.0,
        step_size=0.5
    )

    # 源域数据配置 (48kHz_DE_data)
    source_data_config = []

    # 自动扫描源域数据文件夹
    source_base_path = os.path.join(data_dir, '源域数据集', '48kHz_DE_data')  # 源域数据路径

    if os.path.exists(source_base_path):
        # 扫描各个故障类别文件夹
        fault_folders = {
            'B': 'B',  # 滚动体故障
            'IR': 'IR',  # 内圈故障
            'OR': 'OR'  # 外圈故障
        }

        # 扫描正常状态数据
        normal_folder = os.path.join(source_base_path, '48kHz_Normal_data')
        if os.path.exists(normal_folder):
            for file in os.listdir(normal_folder):
                if file.endswith('.mat'):
                    source_data_config.append({
                        'path': os.path.join(normal_folder, file),
                        'label': 'N'
                    })

        # 扫描故障数据
        for fault_type, folder_name in fault_folders.items():
            fault_folder = os.path.join(source_base_path, folder_name)
            if os.path.exists(fault_folder):
                # 扫描故障尺寸文件夹 (0007, 0014, 0021等)
                for size_folder in os.listdir(fault_folder):
                    size_path = os.path.join(fault_folder, size_folder)
                    if os.path.isdir(size_path):
                        # 对于外圈故障，还需要扫描位置文件夹
                        if fault_type == 'OR':
                            for pos_folder in os.listdir(size_path):
                                pos_path = os.path.join(size_path, pos_folder)
                                if os.path.isdir(pos_path):
                                    for file in os.listdir(pos_path):
                                        if file.endswith('.mat'):
                                            source_data_config.append({
                                                'path': os.path.join(pos_path, file),
                                                'label': fault_type
                                            })
                        else:
                            # 内圈和滚动体故障直接扫描文件
                            for file in os.listdir(size_path):
                                if file.endswith('.mat'):
                                    source_data_config.append({
                                        'path': os.path.join(size_path, file),
                                        'label': fault_type
                                    })

    # 目标域数据配置 (列车轴承数据A-P)
    target_data_config = []
    target_base_path = os.path.join(data_dir, '目标域数据集')   # 修改为您的目标域数据路径

    if os.path.exists(target_base_path):
        for file in os.listdir(target_base_path):
            if file.endswith(('.csv', '.txt', '.mat')) and any(file.startswith(c) for c in 'ABCDEFGHIJKLMNOP'):
                target_data_config.append({
                    'path': os.path.join(target_base_path, file),
                    'label': 'unknown',  # 目标域标签未知
                    'rpm': 600  # 列车轴承转速约600 rpm
                })

    # 合并所有数据配置
    all_data_config = source_data_config + target_data_config

    print(f"找到源域数据文件: {len(source_data_config)} 个")
    print(f"找到目标域数据文件: {len(target_data_config)} 个")
    print(f"总计数据文件: {len(all_data_config)} 个")

    if len(all_data_config) == 0:
        print("未找到任何数据文件，请检查数据路径配置")
        return

    # 处理数据集
    features_df = processor.process_dataset(
        all_data_config,
        output_dir='./extracted_features',
        save_images=True  # 设置为True可保存时频分析图像
    )

    if features_df is not None:
        # 显示特征统计
        print("\n=== 特征提取完成 ===")
        print(
            f"总特征维度: {len([col for col in features_df.columns if col not in ['file', 'sensor', 'segment', 'label', 'fault_size', 'load', 'position', 'rpm', 'bearing_type', 'file_id']])}")
        print(f"总样本数: {len(features_df)}")

        print(f"\n标签分布:")
        print(features_df['label'].value_counts())

        print(f"\n传感器分布:")
        print(features_df['sensor'].value_counts())

        if 'bearing_type' in features_df.columns:
            print(f"\n轴承类型分布:")
            print(features_df['bearing_type'].value_counts())

        # 分别保存源域和目标域特征
        if 'file_id' in features_df.columns:
            # 分离源域和目标域数据
            source_df = features_df[features_df['file_id'].isna()]
            target_df = features_df[~features_df['file_id'].isna()]

            if len(source_df) > 0:
                source_file = os.path.join('./extracted_features', 'source_domain_features.csv')
                source_df.to_csv(source_file, index=False)
                print(f"源域特征已保存: {len(source_df)} 个样本")

            if len(target_df) > 0:
                target_file = os.path.join('./extracted_features', 'target_domain_features.csv')
                target_df.to_csv(target_file, index=False)
                print(f"目标域特征已保存: {len(target_df)} 个样本")

if __name__ == "__main__":
    print(f"当前工作目录: {code_dir}, 项目根目录: {root_dir}, 数据集目录: {data_dir}")
    # 设置工作目录
    os.chdir(root_dir)
    # 运行主程序
    main()