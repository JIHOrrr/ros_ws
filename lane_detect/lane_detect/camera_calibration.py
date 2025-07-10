import cv2
import numpy as np
import glob # 파일 목록을 가져오기 위함

# 1. 체크보드 설정 (사용하는 체크보드의 실제 크기에 맞게 수정)
CHECKERBOARD = (8, 6) # 내부 코너의 개수 (가로-1, 세로-1)
SQUARE_SIZE_MM = 25.0 # 각 사각형의 실제 한 변 길이 (단위는 중요치 않으나 일관성 유지)

# 2. 3D 오브젝트 포인트 준비
# (0,0,0), (1,0,0), (2,0,0) ... 와 같이 3D 좌표를 생성합니다.
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE_MM

# 모든 이미지에 대한 3D 오브젝트 점과 2D 이미지 점을 저장할 리스트
objpoints = [] # 3D point in real world space
imgpoints = [] # 2D points in image plane.

# 3. 캘리브레이션 이미지 로드
images = glob.glob('/home/woong2/opencv/calibration_images/*.jpg') # 이미지를 저장한 폴더 경로

# 이미지 처리 루프
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너를 찾았다면
    if ret == True:
        objpoints.append(objp)

        # 코너 정밀화
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # 코너를 이미지에 그리고 표시 (선택 사항)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(500) # 0.5초 대기 후 다음 이미지

cv2.destroyAllWindows()

# 4. 카메라 캘리브레이션 수행
# 이미지 크기를 가져옵니다 (첫 번째 이미지에서).
# ret, img = cv2.imread(images[0]) # 이 부분을 아래와 같이 수정
img = cv2.imread(images[0]) # cv2.imread는 이미지 데이터만 반환합니다.
height, width = img.shape[:2]
img_size = (width, height)

# calibrateCamera 함수 호출
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# 5. 결과 출력 및 저장
print(f"Reprojection Error: {ret}")
print("Camera Matrix (mtx):\n", mtx)
print("Distortion Coefficients (dist):\n", dist)

# 캘리브레이션 결과 저장
np.savez('camera_calibration_params.npz', mtx=mtx, dist=dist)
print("Calibration parameters saved to camera_calibration_params.npz")

# 왜곡 보정 테스트 (선택 사항)
# 테스트할 이미지 경로를 지정하거나, 캘리브레이션 이미지 중 하나를 사용
test_image_path = '/home/woong2/opencv/calibration_images/calibration_image_002.jpg' # 또는 다른 이미지
test_img = cv2.imread(test_image_path)
h, w = test_img.shape[:2]

# getOptimalNewCameraMatrix를 사용하여 이미지 크롭 없이 왜곡 보정 (필요시)
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted_img = cv2.undistort(test_img, mtx, dist, None, new_camera_mtx)

# 왜곡 보정 후 이미지 표시
cv2.imshow('Original Test Image', test_img)
cv2.imshow('Undistorted Test Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()