"""
Live PA Audio Analyzer - Web Application
Streamlit版

Usage:
    streamlit run pa_analyzer_app.py
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # バックエンドを先に設定
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import io
from pathlib import Path
import tempfile

# matplotlibの設定
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['font.size'] = 10

# ページ設定
st.set_page_config(
    page_title="Live PA Audio Analyzer",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-critical {
        background-color: #ffe6e6;
        padding: 1rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-important {
        background-color: #fff9e6;
        padding: 1rem;
        border-left: 4px solid #ffbb33;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-optional {
        background-color: #e6f7ff;
        padding: 1rem;
        border-left: 4px solid #33b5e5;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitPAAnalyzer:
    """Streamlit用PA解析クラス"""
    
    def __init__(self, audio_file, venue_capacity, stage_volume, pa_system="", notes=""):
        self.audio_file = audio_file
        self.venue_capacity = venue_capacity
        self.stage_volume = stage_volume
        self.pa_system = pa_system
        self.notes = notes
        self.results = {}
        
    def analyze(self):
        """解析実行"""
        try:
            with st.spinner('🎵 音源を読み込んでいます...'):
                # より安全な読み込み方法
                self.y, self.sr = librosa.load(self.audio_file, sr=22050, mono=False, 
                                               duration=300)  # 最大5分まで
                
                if len(self.y.shape) == 1:
                    self.y = np.array([self.y, self.y])
                
                self.y_mono = librosa.to_mono(self.y)
                self.duration = len(self.y_mono) / self.sr
        except Exception as e:
            st.error(f"❌ 音源の読み込みに失敗しました: {str(e)}")
            st.info("💡 対処法: WAV形式に変換するか、短い音源で試してください。")
            raise
        
        # 各解析を実行
        with st.spinner('🔍 ステレオイメージ解析中...'):
            self._analyze_stereo_image()
        
        with st.spinner('📊 ダイナミクス解析中...'):
            self._analyze_dynamics()
        
        with st.spinner('🎼 周波数解析中...'):
            self._analyze_frequency()
        
        with st.spinner('⚡ トランジェント解析中...'):
            self._analyze_transients()
        
        with st.spinner('🔊 低域解析中...'):
            self._analyze_low_end()
        
        return self.results
    
    def _analyze_stereo_image(self):
        """ステレオイメージ解析"""
        left = self.y[0]
        right = self.y[1]
        
        correlation, _ = pearsonr(left, right)
        
        mid = (left + right) / 2
        side = (left - right) / 2
        mid_rms = np.sqrt(np.mean(mid**2))
        side_rms = np.sqrt(np.mean(side**2))
        
        stereo_width = (side_rms / mid_rms * 100) if mid_rms > 0 else 0
        
        self.results['stereo_width'] = stereo_width
        self.results['correlation'] = correlation
        self.results['mid_signal'] = mid
        self.results['side_signal'] = side
    
    def _analyze_dynamics(self):
        """ダイナミクス解析"""
        peak_linear = np.max(np.abs(self.y_mono))
        peak_db = 20 * np.log10(peak_linear) if peak_linear > 0 else -100
        
        rms = np.sqrt(np.mean(self.y_mono**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        
        crest_factor = peak_db - rms_db
        
        hop_length = self.sr // 2
        frame_length = self.sr
        rms_frames = librosa.feature.rms(y=self.y_mono, frame_length=frame_length, 
                                         hop_length=hop_length)[0]
        rms_db_frames = 20 * np.log10(rms_frames + 1e-10)
        
        dynamic_range = np.percentile(rms_db_frames, 95) - np.percentile(rms_db_frames, 5)
        
        self.results['peak_db'] = peak_db
        self.results['rms_db'] = rms_db
        self.results['crest_factor'] = crest_factor
        self.results['dynamic_range'] = dynamic_range
        self.results['rms_frames'] = rms_db_frames
    
    def _analyze_frequency(self):
        """周波数解析"""
        D = np.abs(librosa.stft(self.y_mono))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        avg_spectrum = np.mean(S_db, axis=1)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        bands = [
            (20, 80, "Sub Bass"),
            (80, 250, "Bass"),
            (250, 500, "Low-Mid"),
            (500, 2000, "Mid"),
            (2000, 4000, "High-Mid"),
            (4000, 8000, "Presence"),
            (8000, 16000, "Brilliance"),
        ]
        
        band_energies = []
        for low_freq, high_freq, band_name in bands:
            mask = (freqs >= low_freq) & (freqs < high_freq)
            if np.any(mask):
                band_energy = np.mean(avg_spectrum[mask])
                band_energies.append(band_energy)
            else:
                band_energies.append(-100)
        
        self.results['band_energies'] = band_energies
        self.results['freqs'] = freqs
        self.results['avg_spectrum'] = avg_spectrum
        self.results['bands'] = bands
    
    def _analyze_transients(self):
        """トランジェント解析"""
        onset_env = librosa.onset.onset_strength(y=self.y_mono, sr=self.sr)
        avg_onset_strength = np.mean(onset_env)
        max_onset = np.max(onset_env)
        
        onset_frames = librosa.onset.onset_detect(y=self.y_mono, sr=self.sr, units='frames')
        num_onsets = len(onset_frames)
        onset_density = num_onsets / self.duration
        
        self.results['avg_onset'] = avg_onset_strength
        self.results['max_onset'] = max_onset
        self.results['onset_env'] = onset_env
        self.results['onset_density'] = onset_density
    
    def _analyze_low_end(self):
        """低域解析"""
        nyq = self.sr / 2
        low_cutoff = 40 / nyq
        
        if low_cutoff < 1.0:
            b_low, a_low = signal.butter(4, low_cutoff, btype='lowpass')
            very_low_freq = signal.filtfilt(b_low, a_low, self.y_mono)
            very_low_rms = np.sqrt(np.mean(very_low_freq**2))
        else:
            very_low_rms = 0
        
        if len(self.results.get('band_energies', [])) >= 2:
            sub_bass = self.results['band_energies'][0]
            bass = self.results['band_energies'][1]
            sub_bass_ratio = sub_bass - bass
        else:
            sub_bass_ratio = 0
        
        self.results['very_low_rms'] = very_low_rms
        self.results['sub_bass_ratio'] = sub_bass_ratio
    
    def generate_recommendations(self):
        """改善提案の生成"""
        recommendations = {
            'critical': [],
            'important': [],
            'optional': []
        }
        
        # ステレオイメージ
        stereo_width = self.results.get('stereo_width', 0)
        correlation = self.results.get('correlation', 1)
        
        if correlation < 0.7:
            recommendations['critical'].append({
                'category': 'ステレオイメージ',
                'issue': f'位相相関が低い ({correlation:.3f})',
                'solution': 'Left/Rightチャンネルの位相を確認。パンニングを見直し。',
                'impact': '★★★★★ モノラル互換性が失われている'
            })
        
        if self.venue_capacity < 200 and stereo_width > 30:
            recommendations['important'].append({
                'category': 'ステレオイメージ',
                'issue': f'小規模会場でステレオ幅が広すぎ ({stereo_width:.1f}%)',
                'solution': 'ステレオイメージャーで幅を15%以下に調整。',
                'impact': '★★★ 音像が不安定になりやすい'
            })
        elif self.venue_capacity >= 600 and stereo_width < 25:
            recommendations['important'].append({
                'category': 'ステレオイメージ',
                'issue': f'大規模会場でステレオ幅が狭い ({stereo_width:.1f}%)',
                'solution': 'ステレオイメージャーで幅を30-45%に拡大。',
                'impact': '★★★ 空間表現が不足'
            })
        
        # 音圧・ダイナミクス
        rms_db = self.results.get('rms_db', -100)
        
        if rms_db < -22:
            recommendations['critical'].append({
                'category': '音圧・密度',
                'issue': f'RMSが非常に低い ({rms_db:.1f} dBFS) - 「スカスカ」な音',
                'solution': 'マスターコンプ強化: Threshold -10〜-12dB, Ratio 3:1〜4:1, Attack 20-30ms',
                'impact': '★★★★★ 音圧・密度が決定的に不足'
            })
        elif rms_db < -20:
            recommendations['important'].append({
                'category': '音圧・密度',
                'issue': f'RMSがやや低い ({rms_db:.1f} dBFS)',
                'solution': 'マスターコンプを少し強化。Threshold -12dB, Ratio 2.5:1程度。',
                'impact': '★★★ 音圧感がやや不足'
            })
        
        # 周波数バランス
        if len(self.results.get('band_energies', [])) >= 7:
            mid = self.results['band_energies'][3]
            presence = self.results['band_energies'][5]
            
            if mid - presence > 12:
                if self.stage_volume in ['high', 'medium'] and self.venue_capacity < 200:
                    recommendations['important'].append({
                        'category': '明瞭度',
                        'issue': '高域が弱い（小規模会場・生音あり）',
                        'solution': 'ボーカルchの3-5kHzを選択的に+2〜3dB。',
                        'impact': '★★★ ボーカル明瞭度向上の余地'
                    })
                else:
                    recommendations['critical'].append({
                        'category': '明瞭度',
                        'issue': '高域が大幅に不足',
                        'solution': 'ボーカル3-5kHz +3dB, OH 6-8kHz +2dB, マスター8kHz以上 +1〜2dB',
                        'impact': '★★★★★ 明瞭度が決定的に不足'
                    })
        
        # HPF
        if self.results.get('very_low_rms', 0) > 0.001:
            recommendations['critical'].append({
                'category': 'HPF（システム保護）',
                'issue': '40Hz以下にサブソニック成分あり',
                'solution': 'マスターまたはキック・ベースchにHPF 30-35Hz, 12dB/oct以上を追加',
                'impact': '★★★★ ヘッドルーム損失、システム負荷'
            })
        
        # トランジェント
        avg_onset = self.results.get('avg_onset', 0)
        if avg_onset < 3 and not (self.stage_volume in ['high', 'medium'] and self.venue_capacity < 200):
            recommendations['important'].append({
                'category': 'トランジェント',
                'issue': 'アタック感が不足',
                'solution': 'ドラムchのコンプAttackを遅く（20-30ms）、またはTransientShaperでアタック強調',
                'impact': '★★★ ドラムのパンチ感不足'
            })
        
        return recommendations
    
    def create_visualization(self):
        """グラフ生成"""
        try:
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
            
            # 1. Waveform
            ax1 = fig.add_subplot(gs[0, :])
            time_axis = np.arange(len(self.y_mono)) / self.sr
            ax1.plot(time_axis, self.y_mono, linewidth=0.3, alpha=0.7, color='blue')
            rms_val = 10**(self.results['rms_db']/20)
            ax1.axhline(y=rms_val, color='green', linestyle='--', alpha=0.6, 
                       label=f'RMS: {self.results["rms_db"]:.1f}dB')
            ax1.axhline(y=-rms_val, color='green', linestyle='--', alpha=0.6)
            ax1.set_title('Waveform Overview', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([-1.1, 1.1])
            
            # 2. Frequency Spectrum
            ax2 = fig.add_subplot(gs[1, 0])
            freqs = self.results['freqs'][1:]
            spectrum = self.results['avg_spectrum'][1:]
            ax2.semilogx(freqs, spectrum, linewidth=1.5, color='darkblue')
            ax2.set_title('Frequency Spectrum', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.grid(True, alpha=0.3, which='both')
            ax2.set_xlim([20, self.sr/2])
            
            # 3. Frequency Bands
            ax3 = fig.add_subplot(gs[1, 1])
            band_names = ['Sub\nBass', 'Bass', 'Low\nMid', 'Mid', 'High\nMid', 'Pres', 'Bril']
            colors = ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#F4A460', '#FFA07A', '#FFB6C1']
            ax3.bar(range(len(self.results['band_energies'])), self.results['band_energies'], 
                   color=colors, edgecolor='black', linewidth=1.5)
            ax3.set_xticks(range(len(band_names)))
            ax3.set_xticklabels(band_names, fontsize=9)
            ax3.set_title('Frequency Band Distribution', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Energy (dB)')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Mid/Side
            ax4 = fig.add_subplot(gs[1, 2])
            mid_signal = self.results['mid_signal']
            side_signal = self.results['side_signal']
            time_samples = np.linspace(0, self.duration, min(5000, len(mid_signal)))
            indices = np.linspace(0, len(mid_signal)-1, len(time_samples), dtype=int)
            ax4.plot(time_samples, mid_signal[indices], linewidth=0.8, alpha=0.7, 
                    label='Mid', color='blue')
            ax4.plot(time_samples, side_signal[indices], linewidth=0.8, alpha=0.7, 
                    label='Side', color='red')
            ax4.set_title(f'Mid/Side (Width: {self.results["stereo_width"]:.1f}%)', 
                         fontsize=11, fontweight='bold')
            ax4.set_xlabel('Time (s)')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # 5. RMS Over Time
            ax5 = fig.add_subplot(gs[2, 0])
            hop = self.sr // 2
            time_frames = librosa.frames_to_time(range(len(self.results['rms_frames'])), 
                                                 sr=self.sr, hop_length=hop)
            ax5.plot(time_frames, self.results['rms_frames'], linewidth=1.5, color='green')
            ax5.axhline(y=self.results['rms_db'], color='darkgreen', linestyle='--', 
                       alpha=0.7, label=f'Avg: {self.results["rms_db"]:.1f}dB')
            ax5.set_title('RMS Level Over Time', fontsize=11, fontweight='bold')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('RMS (dBFS)')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([-50, 0])
            
            # 6. Onset Strength
            ax6 = fig.add_subplot(gs[2, 1])
            onset_times = librosa.frames_to_time(range(len(self.results['onset_env'])), sr=self.sr)
            ax6.plot(onset_times, self.results['onset_env'], linewidth=1, color='red', alpha=0.7)
            ax6.axhline(y=self.results['avg_onset'], color='darkred', linestyle='--', 
                       alpha=0.7, label=f'Avg: {self.results["avg_onset"]:.2f}')
            ax6.set_title('Onset Strength', fontsize=11, fontweight='bold')
            ax6.set_xlabel('Time (s)')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            # 7. Spectrogram
            try:
                ax7 = fig.add_subplot(gs[2, 2])
                D = librosa.stft(self.y_mono)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='log',
                                               ax=ax7, cmap='viridis')
                ax7.set_title('Spectrogram', fontsize=11, fontweight='bold')
                ax7.set_ylabel('Frequency (Hz)')
                fig.colorbar(img, ax=ax7, format='%+2.0f dB')
            except Exception as e:
                # Spectrogramが失敗した場合は空のグラフを表示
                ax7 = fig.add_subplot(gs[2, 2])
                ax7.text(0.5, 0.5, 'Spectrogram生成エラー', 
                        ha='center', va='center', transform=ax7.transAxes)
                ax7.set_title('Spectrogram', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"グラフ生成エラー: {str(e)}")
            # エラー時は簡略版のグラフを返す
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'グラフ生成に失敗しました\n{str(e)}', 
                   ha='center', va='center', fontsize=12)
            return fig


def main():
    # ヘッダー
    st.markdown('<h1 class="main-header">🎛️ Live PA Audio Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ライブPA音源の音圧・トランジェント・ステレオイメージ・周波数特性を自動解析</p>', 
                unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 音源アップロード
        uploaded_file = st.file_uploader(
            "音源ファイルをアップロード",
            type=['mp3', 'wav', 'flac', 'm4a'],
            help="PA 2mixの録音ファイル（5分以内、50MB以下推奨）"
        )
        
        # ファイルサイズチェック
        if uploaded_file is not None:
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 100:
                st.error(f"❌ ファイルが大きすぎます（{file_size_mb:.1f}MB）。100MB以下のファイルをアップロードしてください。")
                uploaded_file = None
            else:
                st.success(f"✓ ファイルサイズ: {file_size_mb:.1f}MB")
        
        st.markdown("---")
        
        # 会場情報
        st.subheader("🏛️ 会場情報")
        
        venue_capacity = st.slider(
            "会場キャパシティ（人）",
            min_value=50,
            max_value=2000,
            value=150,
            step=50,
            help="会場の最大収容人数"
        )
        
        stage_volume = st.selectbox(
            "ステージ生音レベル",
            options=['high', 'medium', 'low', 'none'],
            index=1,
            help="ドラム・アンプ等の生音が客席にどれだけ届くか"
        )
        
        pa_system = st.text_input(
            "PAシステム（任意）",
            placeholder="例: d&b V-Series, JBL VTX, etc.",
            help="使用しているPAシステム"
        )
        
        notes = st.text_area(
            "備考（任意）",
            placeholder="バンド編成、会場特性など...",
            help="その他メモ"
        )
        
        st.markdown("---")
        
        analyze_button = st.button(
            "🚀 解析開始",
            type="primary",
            use_container_width=True
        )
    
    # メインコンテンツ
    if uploaded_file is None:
        # 使い方説明
        st.info("👈 左のサイドバーから音源ファイルをアップロードして、解析を開始してください。")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 詳細な解析")
            st.markdown("""
            - ステレオイメージ
            - ダイナミクス・音圧
            - 周波数スペクトラム
            - トランジェント特性
            - 低域詳細分析
            """)
        
        with col2:
            st.markdown("### 🎯 会場規模考慮")
            st.markdown("""
            - 小規模（<200人）
            - 中規模（200-600人）
            - 大規模（>600人）
            - 生音レベルも反映
            """)
        
        with col3:
            st.markdown("### 💡 実践的提案")
            st.markdown("""
            - 🔴 最優先（Critical）
            - 🟡 重要（Important）
            - 🟢 オプション
            - 具体的な設定値
            """)
        
        st.markdown("---")
        
        # サンプル画像（もしあれば）
        st.markdown("### 📈 解析結果サンプル")
        st.info("解析を実行すると、このエリアに詳細なグラフと改善提案が表示されます。")
        
    elif analyze_button:
        # 解析実行
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # 解析実行
            analyzer = StreamlitPAAnalyzer(
                tmp_path,
                venue_capacity,
                stage_volume,
                pa_system,
                notes
            )
            
            results = analyzer.analyze()
            recommendations = analyzer.generate_recommendations()
            
            # 結果表示
            st.success("✅ 解析完了！")
            
            # サマリーメトリクス
            st.markdown("## 📊 解析サマリー")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                stereo_icon = "✓" if 5 <= results['stereo_width'] <= 50 else "⚠️"
                st.metric(
                    "ステレオ幅",
                    f"{results['stereo_width']:.1f}%",
                    delta=stereo_icon,
                    help="Mid/Side信号の比率"
                )
            
            with col2:
                rms_status = "良好" if results['rms_db'] > -20 else "スカスカ" if results['rms_db'] < -22 else "やや弱"
                st.metric(
                    "RMSレベル",
                    f"{results['rms_db']:.1f} dB",
                    delta=rms_status,
                    help="平均音圧レベル"
                )
            
            with col3:
                st.metric(
                    "クレストファクター",
                    f"{results['crest_factor']:.1f} dB",
                    help="ピーク vs RMS"
                )
            
            with col4:
                st.metric(
                    "トランジェント",
                    f"{results['avg_onset']:.2f}",
                    help="アタック強度"
                )
            
            st.markdown("---")
            
            # グラフ表示
            st.markdown("## 📈 詳細グラフ")
            
            with st.spinner('📊 グラフを生成中...'):
                try:
                    fig = analyzer.create_visualization()
                    st.pyplot(fig, use_container_width=True)
                    
                    # ダウンロードボタン
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 グラフをダウンロード",
                        data=buf,
                        file_name="pa_analysis.png",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    plt.close(fig)  # メモリ解放
                    
                except Exception as e:
                    st.error(f"グラフの表示に失敗しました: {str(e)}")
                    st.info("解析結果は正常に完了していますが、グラフの生成でエラーが発生しました。")
                    
                    with st.expander("エラー詳細"):
                        st.exception(e)
            
            st.markdown("---")
            
            # 改善提案
            st.markdown("## 💡 改善提案")
            
            # 会場情報表示
            st.info(f"""
            **会場情報:**  
            キャパ: {venue_capacity}人 | 生音レベル: {stage_volume}  
            {'PAシステム: ' + pa_system if pa_system else ''}
            """)
            
            # Critical
            if recommendations['critical']:
                st.markdown("### 🔴 最優先（すぐに対処すべき）")
                for rec in recommendations['critical']:
                    st.markdown(f"""
                    <div class="recommendation-critical">
                        <strong>{rec['category']}</strong><br>
                        <strong>問題:</strong> {rec['issue']}<br>
                        <strong>対策:</strong> {rec['solution']}<br>
                        <strong>影響度:</strong> {rec['impact']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🔴 最優先項目: なし（優秀！）")
            
            # Important
            if recommendations['important']:
                st.markdown("### 🟡 重要（改善推奨）")
                for rec in recommendations['important']:
                    st.markdown(f"""
                    <div class="recommendation-important">
                        <strong>{rec['category']}</strong><br>
                        <strong>問題:</strong> {rec['issue']}<br>
                        <strong>対策:</strong> {rec['solution']}<br>
                        <strong>影響度:</strong> {rec['impact']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("🟡 重要項目: なし（良好！）")
            
            # Optional
            if recommendations['optional']:
                st.markdown("### 🟢 オプション（余裕があれば）")
                for rec in recommendations['optional']:
                    st.markdown(f"""
                    <div class="recommendation-optional">
                        <strong>{rec['category']}</strong><br>
                        <strong>問題:</strong> {rec['issue']}<br>
                        <strong>対策:</strong> {rec['solution']}<br>
                        <strong>影響度:</strong> {rec['impact']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🟢 オプション項目: なし")
            
            # 詳細データ（Expander）
            with st.expander("📋 詳細データを表示"):
                st.json({
                    "stereo_width": f"{results['stereo_width']:.2f}%",
                    "correlation": f"{results['correlation']:.4f}",
                    "peak_db": f"{results['peak_db']:.2f} dBFS",
                    "rms_db": f"{results['rms_db']:.2f} dBFS",
                    "crest_factor": f"{results['crest_factor']:.2f} dB",
                    "dynamic_range": f"{results['dynamic_range']:.2f} dB",
                    "avg_onset": f"{results['avg_onset']:.3f}",
                    "onset_density": f"{results['onset_density']:.2f} /sec",
                    "very_low_rms": f"{results['very_low_rms']:.6f}",
                    "sub_bass_ratio": f"{results['sub_bass_ratio']:.2f} dB"
                })
            
        except Exception as e:
            st.error(f"❌ エラーが発生しました")
            
            # エラーの種類を判定
            error_msg = str(e)
            
            if "soundfile" in error_msg.lower() or "audioread" in error_msg.lower():
                st.error("**原因**: 音源ファイルの読み込みに必要なライブラリが不足しています")
                st.info("""
                **解決方法**:
                
                ターミナルで以下を実行してください:
                ```bash
                pip install soundfile audioread
                ```
                
                その後、アプリを再起動してください。
                """)
            elif "memory" in error_msg.lower():
                st.error("**原因**: メモリ不足")
                st.info("**解決方法**: より短い音源ファイル（1-2分）で試してください")
            else:
                st.error(f"**エラー詳細**: {error_msg}")
                st.info("""
                **対処法**:
                1. WAV形式の音源で試す
                2. より短い音源（1-2分）で試す
                3. ファイルサイズを小さくする
                
                それでも解決しない場合は、エラーメッセージをコピーしてフィードバックください。
                """)
            
            # 詳細なエラー情報（開発者向け）
            with st.expander("🔧 詳細なエラー情報（開発者向け）"):
                st.exception(e)
        
        finally:
            # 一時ファイル削除
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🎛️ Live PA Audio Analyzer v2.0</p>
        <p>作成: 石垣（PAエンジニア） | <a href="https://note.com/your-profile">note</a> | <a href="https://twitter.com/your-handle">X</a></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
