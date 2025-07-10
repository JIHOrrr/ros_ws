import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform
import os

# 한글 폰트 설정 (나눔고딕)
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
else:
    print("[경고] 나눔고딕 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 경로 및 저장 디렉토리
csv_path = '/home/jiho/robot_ws/src/turtlebot_tracking/tracking_stats.csv'
save_dir = '/home/jiho/robot_ws/src/turtlebot_tracking/visualizations/'
os.makedirs(save_dir, exist_ok=True)

# 데이터 불러오기 및 결측치 제거
df = pd.read_csv(csv_path)
df_clean = df.dropna()

# 탐지 여부에 따른 통계 분석용 컬럼 추가
df_clean['detected'] = df_clean['is_true_detection'] == 1

# =======================
# 1. 탐지 여부 막대그래프
# =======================
plt.figure(figsize=(6,4))
sns.countplot(data=df_clean, x='detected', palette='Set2')
plt.xticks([0, 1], ['탐지 실패', '탐지 성공'])
plt.title('탐지 성공/실패 프레임 수')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detection_count.png'))
plt.show()

# =======================
# 2. 프레임별 탐지 성공 여부 시계열
# =======================
plt.figure(figsize=(14,4))
plt.plot(df_clean['frame'], df_clean['is_true_detection'], drawstyle='steps-post', color='blue')
plt.title('프레임별 탐지 성공 여부')
plt.xlabel('프레임')
plt.ylabel('탐지 (1=성공, 0=실패)')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detection_timeline.png'))
plt.show()

# =======================
# 3. 탐지 지속 시간 분포 (연속 성공 프레임 길이)
# =======================
from itertools import groupby
from operator import itemgetter

# 탐지 성공 구간의 길이 측정
detected_frames = df_clean[df_clean['is_true_detection'] == 1]['frame'].tolist()
groups = [list(g) for k, g in groupby(enumerate(detected_frames), lambda x: x[0]-x[1])]
streaks = [len([x for _, x in group]) for group in groups]

plt.figure(figsize=(8,6))
sns.histplot(streaks, bins=range(1, max(streaks)+2), color='green')
plt.title('연속 탐지 성공 지속 시간 분포')
plt.xlabel('연속 탐지된 프레임 수')
plt.ylabel('빈도')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'detection_streaks.png'))
plt.show()

# =======================
# 4. 탐지 ID 빈도수 시각화
# =======================
plt.figure(figsize=(10,4))
sns.countplot(data=df_clean[df_clean['track_id'] != -1], x='track_id', order=df_clean['track_id'].value_counts().index, palette='tab10')
plt.title('추적된 ID별 등장 빈도')
plt.xlabel('Track ID')
plt.ylabel('프레임 수')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'track_id_frequency.png'))
plt.show()

print("✅ 시각화 완료. 결과 저장 위치:", save_dir)
