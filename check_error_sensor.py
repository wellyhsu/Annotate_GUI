import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os
import trimesh
from scipy.spatial import procrustes
from typing import Tuple, List

# Master <-> Slave一組外參數
# Master <-> 子一相機一組外參數

# 要改成讀取RGB資料夾下所有圖片(0~449)
# === 新的路徑設定 ===
Object = 'Dropper'
M_image_filepath     = fr"./Data/{Object}/D435f_Master"
S_image_filepath     = fr"./Data/{Object}/Sensor"
M_annotate_filepath  = fr"./Annotate/{Object}/D435f_Master"
S_annotate_filepath  = fr"./Annotate/{Object}/Sensor"

# === 手動設定（不用 argparse）===
START_IDX   = 0   # 從排序後清單的第 120 張開始（0-based）
MMPJPE_THR  = 15.0  # MPJPE 門檻（mm），超過就印出檔名與數值
Show_all = True

Depth_mode = 'Depth_filter' # 'Depth_origin'
# === 自動取得 hand_mode (從第一個 _kpts_2d_glob_*.npy 檔案 in Master Annotate) ===
print(os.path.join(M_annotate_filepath, "*_kpts_2d_glob_*.npy"))
first_kpt2d_file = sorted(glob.glob(os.path.join(M_annotate_filepath, "*_kpts_2d_glob_*.npy")))[0]
hand_mode = os.path.splitext(first_kpt2d_file)[0].split("_")[-1]
if hand_mode == "right":
    hand_mode="r"
elif hand_mode== "left":
    hand_mode="l"
print("偵測到 hand_mode =", hand_mode)

# === 找出所有 Master/Sensor RGB 圖片，依照數字排序 ===
M_images = sorted(glob.glob(os.path.join(M_image_filepath, "RGB", "*.png")),
                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
S_images = sorted(glob.glob(os.path.join(S_image_filepath, "*.png")),
                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                  
# >>> 只取出 Master 的偶數編號 (0, 2, 4, ..., 448) <<<
M_images = [p for p in M_images
            if int(os.path.splitext(os.path.basename(p))[0]) % 2 == 0]
S_images = [p for p in S_images
            if int(os.path.splitext(os.path.basename(p))[0]) % 2 == 0]
# 套用起始索引（基本邊界檢查）
if not (0 <= START_IDX <= len(M_images)):
    raise ValueError(f"START_IDX={START_IDX} 超出範圍（共有 {len(M_images)} 張）")

M_images = M_images[START_IDX:]
S_images = S_images[START_IDX:]

# === 確保兩邊數量一致 ===
print(f"[INFO] Master 偶數張數量：{len(M_images)}，Sensor 張數量：{len(S_images)}")
assert len(M_images) == len(S_images), "Master 與 Sensor 檔案數量不一致！"

print(f"[INFO] 從第 {START_IDX} 張開始（本次會處理 {len(M_images)} 張）")

def visualize_depth_points(depth_image, cam2_to_cam1_points_2d, window_name="depth_filter Visualization"):
        # 複製一份影像來畫點（轉換為 3-channel 方便畫顏色）
        vis_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

        for i, (x, y) in enumerate(cam2_to_cam1_points_2d):
            x_int = int(round(x))
            y_int = int(round(y))

            # 邊界檢查
            if 0 <= x_int < width and 0 <= y_int < height:
                depth = depth_image[y_int, x_int]
                # 畫圓
                cv2.circle(vis_image, (x_int, y_int), 5, (0, 0, 255), -1)
                # 顯示深度文字
                cv2.putText(vis_image, f"{depth}", (x_int + 5, y_int - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # cv2.imshow(window_name, vis_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

# 提取每個2D點對應的深度值
def Find_min_depth(cam_intel, depth_image):
        depth_values = []

        for (x, y) in cam_intel:
            x_int, y_int = int(round(x)), int(round(y))
            depth = depth_image[y_int, x_int]
            depth_values.append(depth)

        depth_values = np.array(depth_values)

        # 找出最小深度值與對應index
        min_index = np.argmin(depth_values)
        min_depth = depth_values[min_index]
        # print(depth_values)
        return depth_values, min_index, min_depth

def find_nearest_hand_point(depth_values, points_2d, segmentation):
        """
        depth_values : (21,) 深度值 (float 或 int)
        points_2d    : (21, 2) -> [[x0, y0], [x1, y1], ...] 已是 pixel 座標
        segmentation : (H, W) 0/1/2 mask，1 代表手

        回傳 (best_index, best_depth)；若皆無符合條件，回傳 (None, None)
        """
        # 依深度值由小到大排序索引值
        order = np.argsort(depth_values)          # 離相機最近的排在最前面
        h, w = segmentation.shape

        for idx in order:
            x, y = map(int, np.round(points_2d[idx]))  # 四捨五入取整
            # 邊界檢查
            if 0 <= x < w and 0 <= y < h:
                if segmentation[y, x] == 1:           # 只要落在「手」區域
                    return idx, depth_values[idx]
                return idx, depth_values[idx]
        # 21 個都沒有落在 segmentation==1
        return None, None

def project_3d_to_2d(kpts_3d, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    kpts_2d = []
    for X, Y, Z in kpts_3d:
        if np.isnan(Z) or Z == 0:
            kpts_2d.append([np.nan, np.nan])
        else:
            x = fx * X / Z + cx
            y = fy * Y / Z + cy
            kpts_2d.append([x, y])
    return np.array(kpts_2d)

def reprojection_error(kpts_proj, kpts_2d):
    valid = ~np.isnan(kpts_proj).any(axis=1) & ~np.isnan(kpts_2d).any(axis=1)
    return np.mean(np.linalg.norm(kpts_proj[valid] - kpts_2d[valid], axis=1))

def draw_keypoints(
    image: np.ndarray,
    kpts_2d: np.ndarray,
    color: Tuple[int, int, int],
    radius: int = 4
) -> np.ndarray:
    """
    在影像上畫出 2D keypoints。
    - image  : BGR 影像 (np.uint8)
    - kpts_2d: (N,2) array, 可能含 nan
    - color  : (B, G, R)
    """
    h, w = image.shape[:2]

    vis_img = image.copy()
    for x, y in kpts_2d:
        if np.isnan(x) or np.isnan(y):
            continue
        x_int, y_int = int(round(x)), int(round(y))
        if 0 <= x_int < w and 0 <= y_int < h:
            cv2.circle(vis_img, (x_int, y_int), radius, color, -1)
    return vis_img

def draw_skeleton(
    image: np.ndarray,
    kpts_2d: np.ndarray,
    skeleton: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = 2
) -> np.ndarray:
    """
    根據骨架連線在影像上畫線段。

    - image     : 原始 BGR 影像
    - kpts_2d   : shape=(21, 2)，keypoints 位置
    - skeleton  : 連線索引對，例：[(0,1), (1,2), ...]
    - color     : (B, G, R)
    - thickness : 線寬
    """
    vis_img = image.copy()
    h, w = vis_img.shape[:2]

    for i, j in skeleton:
        x1, y1 = kpts_2d[i]
        x2, y2 = kpts_2d[j]

        if np.isnan([x1, y1, x2, y2]).any():
            continue

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        if (0 <= p1[0] < w and 0 <= p1[1] < h and
            0 <= p2[0] < w and 0 <= p2[1] < h):
            cv2.line(vis_img, p1, p2, color, thickness)

    return vis_img

def fuse_3d_keypoints(kpts_cam1, kpts_cam2, vis1, vis2):
    fused = []
    for k1, k2, v1, v2 in zip(kpts_cam1, kpts_cam2, vis1, vis2):
        if v1 and v2:
            fused.append((k1 + k2) / 2)
        elif v1:
            fused.append(k1)
        elif v2:
            fused.append(k2)
        else:
            fused.append((k1 + k2) / 2)
    return np.array(fused)

def pad_mesh_along_rays(mesh: np.ndarray,
                        K: np.ndarray,
                        depth_offset: float) -> np.ndarray:
    """
    mesh: (N,3) 原始 mesh 頂點（相機座標系）
    K:     (3,3) 相機內參
    depth_offset: 要加到每個點的深度偏移量（mm），例如 min_depth_cam - min_z

    回傳 shape (N,3) 的新頂點陣列
    """
    # 1) 投影到像素平面（齊次座標）
    uvh = (K @ mesh.T).T               # shape (N,3)
    uv  = uvh[:, :2] / uvh[:, 2:3]     # shape (N,2)

    # 2) 建構齊次像素座標 [u, v, 1]
    uv1 = np.concatenate([uv, np.ones((uv.shape[0],1))], axis=1)  # (N,3)

    # 3) 計算光線方向（可選擇是否正規化，後續會乘上深度）
    K_inv = np.linalg.inv(K)
    dirs  = (K_inv @ uv1.T).T  # (N,3)

    # 4) 新深度 = 原始深度 + 偏移量
    new_depth = mesh[:,2] + depth_offset  # (N,)

    # 5) 沿著光線重建 3D 點
    return dirs * new_depth[:, None]      # 廣播 → (N,3)

def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t

def umeyama_weighted(P, Q, weights=None):
    """
    P, Q: n x 3
    weights: n 長度向量
    """
    assert P.shape == Q.shape
    n, dim = P.shape
    if weights is None:
        weights = np.ones(n)

    Wsum = np.sum(weights)
    mean_P = np.sum(P * weights[:, None], axis=0) / Wsum
    mean_Q = np.sum(Q * weights[:, None], axis=0) / Wsum

    centeredP = P - mean_P
    centeredQ = Q - mean_Q

    C = (centeredP.T * weights).dot(centeredQ) / Wsum

    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = V.dot(Wt)
    varP = np.sum(weights * np.sum(centeredP**2, axis=1)) / Wsum
    c = np.sum(S) / varP

    t = mean_Q - mean_P.dot(c*R)
    return c, R, t

def align_hand(P, Q, important_indices=[0,5,9,13,17], threshold=0.000001):
    """
    P: source points (n x 3)
    Q: target points (n x 3)
    important_indices: 優先對齊的關節
    threshold: 當這些點距離誤差小於 threshold 時，才再對齊剩餘點
    """
    n = P.shape[0]
    
    # Step1: 先對齊重要點
    weights = np.zeros(n)
    weights[important_indices] = 1.0  # 重要點權重1
    c1, R1, t1 = umeyama_weighted(P, Q, weights)

    P_aligned = P.dot(c1*R1) + t1

    # Step2: 計算重要點對齊誤差
    error = np.linalg.norm(P_aligned[important_indices] - Q[important_indices], axis=1)
    # print("error:", error)
    if np.all(error < threshold):
        # 若誤差小於門檻，再對齊剩餘點
        remaining_indices = [i for i in range(n) if i not in important_indices]
        if remaining_indices:
            weights = np.zeros(n)
            weights[remaining_indices] = 1.0
            c2, R2, t2 = umeyama_weighted(P_aligned, Q, weights)
            P_aligned = P_aligned.dot(c2*R2) + t2
            # 合併兩次 scale & rotation (可選，這裡直接疊加)
    return P_aligned, c1, R1, t1

def compute_pa_mpjpe(pred, gt):
    error = np.linalg.norm(pred - gt, axis=1)  # [K]
    return np.mean(error)

def read_faces_from_obj(file_path):
    """
    從 .obj 文件中讀取 faces 資訊
    :param file_path: .obj 檔案路徑
    :return: faces（numpy array），形狀為 (M, 3)，每行是三個頂點的索引
    """
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('f '):  # .obj 中的面是以 'f ' 開頭
                # 提取面索引，並將索引轉換為整數（注意：obj 的索引是從 1 開始的，需要減 1）
                face = [int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]]
                faces.append(face)
    
    return np.array(faces)



for M_img_path, S_img_path in zip(M_images, S_images):
    name = os.path.splitext(os.path.basename(M_img_path))[0]  # e.g. "52"
    print("name: ", name)
    # 影像
    M_image = cv2.imread(M_img_path)
    S_image = cv2.imread(S_img_path)
    height, width = M_image.shape[:2]
    S_height, S_width = S_image.shape[:2]
    # print(f"RGB image: {width}*{height}")

    # 深度 (來自 Annotate 路徑)
    depth_image_cam1 = cv2.imread(os.path.join(M_image_filepath, Depth_mode, f"{name}.png"), cv2.IMREAD_UNCHANGED)
    D_height, D_width = depth_image_cam1.shape[:2]
    if (height != D_height) or (width != D_width):
        # 使用插值技術進行放大（恢復原始尺寸）
        depth_image_cam1 = cv2.resize(depth_image_cam1, (width, height), interpolation=cv2.INTER_LINEAR)

    # print(f"Depth image: {width}*{height}")
    # Keypoints / Mesh (來自 Annotate 路徑)
    Kpt_2d_filename = f"{name}_kpts_2d_glob_{hand_mode}.npy"
    Kpt_3d_filename = f"{name}_kpts_3d_glob_{hand_mode}.npy"
    mesh_filename   = f"{name}_mesh_vert_3d_glob_{hand_mode}.npy"

    # 建立完整路徑
    M_kpt2d_path = os.path.join(M_annotate_filepath, Kpt_2d_filename)
    M_kpt3d_path = os.path.join(M_annotate_filepath, Kpt_3d_filename)
    M_mesh_path  = os.path.join(M_annotate_filepath, mesh_filename)
    S_kpt2d_path = os.path.join(S_annotate_filepath, Kpt_2d_filename)
    S_kpt3d_path = os.path.join(S_annotate_filepath, Kpt_3d_filename)
    S_mesh_path  = os.path.join(S_annotate_filepath, mesh_filename)
    # 檔案不存在就跳過
    if not (os.path.exists(M_kpt2d_path) and os.path.exists(M_kpt3d_path) and os.path.exists(M_mesh_path) and os.path.exists(S_kpt2d_path) and os.path.exists(S_kpt3d_path) and os.path.exists(S_mesh_path)):
        print(f"缺少 {name} 的標註檔，跳過")
        break
    points_2d_cam1 = np.load(M_kpt2d_path)
    points_2d_cam2 = np.load(S_kpt2d_path)
    points_2d_cam1 = points_2d_cam1[:, [1, 0]]
    points_2d_cam2 = points_2d_cam2[:, [1, 0]]

    points_3d_cam1 = np.load(M_kpt3d_path)*10  # 單位mm  
    points_3d_cam2 = np.load(S_kpt3d_path)*10  # 單位mm  
    # 交換 x 和 y 座標
    points_3d_cam1[:, [0, 1]] = points_3d_cam1[:, [1, 0]]
    points_3d_cam2[:, [0, 1]] = points_3d_cam2[:, [1, 0]]

    mesh_cam1      = np.load(M_mesh_path)
    mesh_cam2      = np.load(S_mesh_path)
    
    faces_cam1 = read_faces_from_obj(os.path.join(M_annotate_filepath, f"{name}_mano_mesh_{hand_mode}.obj"))
    faces_cam2 = read_faces_from_obj(os.path.join(S_annotate_filepath, f"{name}_mano_mesh_{hand_mode}.obj"))
      

    # 若 hand_mode 是左手，將座標轉換成左手座標系統
    if hand_mode == "l":
        # 左右翻轉 2D keypoints (x軸翻轉)
        points_2d_cam1[:, 0] = 1.0 - points_2d_cam1[:, 0]
        points_2d_cam2[:, 0] = 1.0 - points_2d_cam2[:, 0]

        # 左右翻轉 3D keypoints (x軸翻轉)
        points_3d_cam1[:, 0] = points_3d_cam1[:, 0] * -1  # 翻轉 x 坐標
        points_3d_cam2[:, 0] = points_3d_cam2[:, 0] * -1  # 翻轉 x 坐標

        # 左右翻轉 Mesh (x軸翻轉)
        mesh_cam1[:, 0] = mesh_cam1[:, 0] * -1  # 翻轉 x 坐標
        mesh_cam2[:, 0] = mesh_cam2[:, 0] * -1  # 翻轉 x 坐標

    print(f"已讀取 {name}.png")

    # Sensor
    mtxL= np.array([ 
        [123.30972927,   0.        , 60.71911309],
       [  0.        , 124.36198841, 64.57786929],
       [  0.        ,   0.        ,  1.        ]
    ])
    distL = np.array( [-2.12126306e-01, -5.31342339e-01,  1.08729770e-03,  5.72094100e-04, 1.09756130e+00])

    # D435f Master
    mtxR = np.array([
        [609.213254,   0.       , 318.25896107],
       [  0.       , 604.67951874, 235.85896107],
       [  0.       ,   0.       ,   1.        ]
    ]) 
    distR = np.array([ 1.78210798e-02,  6.98242976e-01, -1.43163258e-04,  6.94916940e-03, -2.46461169e+00])




    R = np.array([
        [ 1.0, 0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [0.0,   0.0,  1.]])
    T=np.array([
    [ 0],
    [ 0.0],
    [ 0]
    ])

    T = T.reshape(1, 3)  # 將平移向量變成 1x3

    # =============================================================================================================================================================
    # === 6. 使用 R 和 T 對齊 3D 手部 Keypoints =====================================================================================================================
    # =============================================================================================================================================================
    # 從 Camera 1 到 Camera 2 的變換：
    # Cam2_point = R × Cam1_point + T
    # 要從 Cam2 轉到 Cam1：
    # Cam1_point = R.T × (Cam2_point - T)

    # K_cam = np.array([
    #     [571.2, 0, 320],
    #     [0, 571.2, 240],
    #     [0, 0, 1]
    # ])
    K_cam_M = mtxR
    K_cam_S = mtxL


    # if Kpt_2d_filename.endswith('_l.npy'):
    #     M_image = cv2.flip(M_image, 1)  # 左右翻轉
    #     S_image = cv2.flip(S_image, 1)  # 左右翻轉
    #     depth_image_cam1 = cv2.flip(depth_image_cam1, 1)  # 左右翻轉
    #     depth_image_cam2 = cv2.flip(depth_image_cam2, 1)  # 左右翻轉
    visualize_depth_points(depth_image_cam1, points_2d_cam1)


    # 整體深度對齊：使所有 z 移到對齊最小的 z 值
    points_3d_cam1_padding=points_3d_cam1.copy()
    points_3d_cam2_padding=points_3d_cam2.copy()


    # 將 cam2 座標(右)轉到 cam1 座標(左)系（或世界座標）
    # Sensor view座標轉到Master view座標
    points_cam2_to_cam1 = (points_3d_cam2 @ R) + T # (N, 3) 
    points_cam2_to_cam1_padding = (points_3d_cam2_padding @ R) + T # (N, 3) 

    # Master view座標轉到Sensor view座標
    points_cam1_to_cam2 = (points_3d_cam1 - T )@ R.T  # (N, 3) 
    points_cam1_to_cam2_padding = (points_3d_cam1_padding - T) @ R.T  # (N, 3) 


    # ========================  去除旋轉平移、對齊手腕  ==============================================================================================================

    # # 2 -> 1
    # c1, R1, T1 = umeyama(points_cam2_to_cam1_padding, points_3d_cam1_padding)
    # points_cam2_to_cam1_padding = points_cam2_to_cam1_padding.dot(c1*R1) + T1

    # # 1-> 2
    # c2, R2, T2 = umeyama(points_cam1_to_cam2_padding, points_3d_cam2_padding)
    # points_cam1_to_cam2_padding = points_cam1_to_cam2_padding.dot(c2*R2) + T2

    # 2 -> 1
    points_cam2_to_cam1_padding, c1, R1, T1 = align_hand(points_cam2_to_cam1_padding, points_3d_cam1_padding)
    # 1-> 2
    points_cam1_to_cam2_padding, c2, R2, T2 = align_hand(points_cam1_to_cam2_padding, points_3d_cam2_padding)


    # 定義 MediaPipe 手部骨架連線
    skeleton = [
        [0,1],[1,2],[2,3],[3,4],        # Thumb
        [0,5],[5,6],[6,7],[7,8],        # Index
        [0,9],[9,10],[10,11],[11,12],   # Middle
        [0,13],[13,14],[14,15],[15,16], # Ring
        [0,17],[17,18],[18,19],[19,20]  # Pinky
    ]


    # =============== (相加除以2) 加權融合 =====================================================

    vis1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
    vis2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

    # # manual
    # Cam1_manual_fuse_3D_keypoints = fuse_3d_keypoints(points_3d_cam1_padding, points_cam2_to_cam1_padding, vis1, vis2)
    # Cam2_manual_fuse_3D_keypoints = fuse_3d_keypoints(points_3d_cam2_padding, points_cam1_to_cam2_padding, vis1, vis2)

    # c1, R1, T1 = umeyama(Cam1_manual_fuse_3D_keypoints, points_3d_cam1_padding)
    # Cam1_manual_fuse_3D_keypoints = Cam1_manual_fuse_3D_keypoints.dot(c1*R1) + T1
    # c1, R1, T1 = umeyama(Cam2_manual_fuse_3D_keypoints, points_3d_cam2_padding)
    # Cam2_manual_fuse_3D_keypoints = Cam2_manual_fuse_3D_keypoints.dot(c1*R1) + T1


    Mean = (compute_pa_mpjpe(points_cam2_to_cam1_padding, points_3d_cam1_padding) + compute_pa_mpjpe(points_cam1_to_cam2_padding, points_3d_cam2_padding))/2
    if Show_all == True or Mean > 15:
        print("Master view and Sensor view MPJPE:", Mean, "mm")

    # print("============ Verify ===========")
    # Cam1_manual_fuse_kpts_proj = project_3d_to_2d(Cam1_manual_fuse_3D_keypoints, K_cam_M)
    # Cam2_manual_fuse_kpts_proj = project_3d_to_2d(Cam2_manual_fuse_3D_keypoints, K_cam_M)

    # print("intel Cam1 reprojection error: ", reprojection_error(cam1_intel_kpts_proj, points_2d_cam1),"pixels")
    # print("Fuse Cam1 reprojection error: ", reprojection_error(Cam1_kpts_proj, points_2d_cam1),"pixels")
    # print("Fuse Cam1toCam2 reprojection error: ", reprojection_error(Cam1toCam2_kpts_proj, points_2d_cam2),"pixels")

    # print("intel Cam2 reprojection error: ", reprojection_error(cam2_intel_kpts_proj, points_2d_cam2),"pixels")
    # print("Fuse Cam2 reprojection error: ", reprojection_error(Cam2_kpts_proj, points_2d_cam2),"pixels")
    # print("Fuse Cam2toCam1 reprojection error: ", reprojection_error(Cam2toCam1_kpts_proj, points_2d_cam1),"pixels")

    if Show_all == True or Mean > 15:
        fig = plt.figure(figsize=(10, 6))

        # 第三張圖：Master + Transformed Sensor
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        ax3.set_title("Master View Fuse")
        # ax3.scatter(Cam1_manual_fuse_3D_keypoints[:, 0], Cam1_manual_fuse_3D_keypoints[:, 1], Cam1_manual_fuse_3D_keypoints[:, 2], c="red", label="manual fuse Keypoints") # 記得改回Camera 2 Keypoints
        ax3.scatter(points_cam2_to_cam1_padding[:, 0], points_cam2_to_cam1_padding[:, 1], points_cam2_to_cam1_padding[:, 2], c="blue", label="Sensor to Master Coord") # 記得改回Camera 2 Keypoints
        ax3.scatter(points_3d_cam1_padding[:, 0], points_3d_cam1_padding[:, 1], points_3d_cam1_padding[:, 2], c="purple", label="Master Coord") # 記得改回Camera 2 Keypoints
        # 畫骨架：Master 為藍線，Sensor 為紅線，intel purple
        for i, j in skeleton:
            # ax3.plot([Cam1_manual_fuse_3D_keypoints[i, 0], Cam1_manual_fuse_3D_keypoints[j, 0]],
            #         [Cam1_manual_fuse_3D_keypoints[i, 1], Cam1_manual_fuse_3D_keypoints[j, 1]],
            #         [Cam1_manual_fuse_3D_keypoints[i, 2], Cam1_manual_fuse_3D_keypoints[j, 2]], c='red')
            ax3.plot([points_cam2_to_cam1_padding[i, 0], points_cam2_to_cam1_padding[j, 0]],
                    [points_cam2_to_cam1_padding[i, 1], points_cam2_to_cam1_padding[j, 1]],
                    [points_cam2_to_cam1_padding[i, 2], points_cam2_to_cam1_padding[j, 2]], c='blue')
            ax3.plot([points_3d_cam1_padding[i, 0], points_3d_cam1_padding[j, 0]],
                    [points_3d_cam1_padding[i, 1], points_3d_cam1_padding[j, 1]],
                    [points_3d_cam1_padding[i, 2], points_3d_cam1_padding[j, 2]], c='purple')
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.legend()

        # 第四張圖：Sensor + Transformed Master
        ax4 = fig.add_subplot(1, 2, 2, projection="3d")
        ax4.set_title("Sensor View Fuse")
        # ax4.scatter(Cam2_manual_fuse_3D_keypoints[:, 0], Cam2_manual_fuse_3D_keypoints[:, 1], Cam2_manual_fuse_3D_keypoints[:, 2], c="r", label="manual fuse Keypoints")
        ax4.scatter(points_cam1_to_cam2_padding[:, 0], points_cam1_to_cam2_padding[:, 1], points_cam1_to_cam2_padding[:, 2], c="purple", label="Master to Sensor Coord")
        ax4.scatter(points_3d_cam2_padding[:, 0], points_3d_cam2_padding[:, 1], points_3d_cam2_padding[:, 2], c="blue", label="Sensor Coord")

        # 畫骨架：Master 為藍線，Sensor 為紅線，intel purple
        for i, j in skeleton:
            # ax4.plot([Cam2_manual_fuse_3D_keypoints[i, 0], Cam2_manual_fuse_3D_keypoints[j, 0]],
            #         [Cam2_manual_fuse_3D_keypoints[i, 1], Cam2_manual_fuse_3D_keypoints[j, 1]],
            #         [Cam2_manual_fuse_3D_keypoints[i, 2], Cam2_manual_fuse_3D_keypoints[j, 2]], c='red')
            ax4.plot([points_cam1_to_cam2_padding[i, 0], points_cam1_to_cam2_padding[j, 0]],
                    [points_cam1_to_cam2_padding[i, 1], points_cam1_to_cam2_padding[j, 1]],
                    [points_cam1_to_cam2_padding[i, 2], points_cam1_to_cam2_padding[j, 2]], c='purple')
            ax4.plot([points_3d_cam2_padding[i, 0], points_3d_cam2_padding[j, 0]],
                    [points_3d_cam2_padding[i, 1], points_3d_cam2_padding[j, 1]],
                    [points_3d_cam2_padding[i, 2], points_3d_cam2_padding[j, 2]], c='blue')
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.set_zlabel("Z")
        ax4.legend()

        plt.tight_layout()
        plt.show()


    # ── Master RGB 影像 ───────────────────────────────────────────
    master_vis = M_image.copy()
    points_2d_cam1[:, 0] *= width
    points_2d_cam1[:, 1] *= height
    scale = 512/126
    K_cam_S = mtxL.copy().astype(np.float64)
    K_cam_S[0,0] *= scale  # fx
    K_cam_S[1,1] *= scale  # fy
    K_cam_S[0,2] *= scale  # cx
    K_cam_S[1,2] *= scale  # cy
    points_2d_SM= project_3d_to_2d(points_cam2_to_cam1_padding, K_cam_M)
    points_2d_MS = project_3d_to_2d(points_cam1_to_cam2_padding, K_cam_S)

    # 先畫骨架線
    # print(points_2d_cam1)
    master_vis = draw_skeleton(master_vis, points_2d_cam1, skeleton, (255, 0, 255))   # 紫色：Groundtruth
    master_vis = draw_skeleton(master_vis, points_2d_SM, skeleton, (255, 0, 0)) # 藍色：M_Fuse
    # master_vis = draw_skeleton(master_vis, Cam2toCam1_kpts_proj, skeleton, (0, 255, 255)) # 黃色：M->S Fuse
    # 再畫 keypoints 點
    master_vis = draw_keypoints(master_vis, points_2d_cam1, (255, 0, 255))   # 紫色：Groundtruth
    master_vis = draw_keypoints(master_vis, points_2d_SM, (255, 0, 0)) # 紅色：M_Fuse
    # master_vis = draw_keypoints(master_vis, Cam2toCam1_kpts_proj, (0, 255, 255)) # 黃色：M->S Fuse
    if Show_all == True or Mean > 15:
        cv2.imshow("Master view(blue=Sensor, purple=Master)", master_vis)

    # ── Sensor RGB 影像 ────────────────────────────────────────────
    points_2d_cam2[:, 0] *= S_width
    points_2d_cam2[:, 1] *= S_height
    sensor_vis  = S_image.copy()
    sensor_vis = cv2.resize(sensor_vis, (512, 512), interpolation=cv2.INTER_LINEAR)
    # 先畫骨架線
    sensor_vis = draw_skeleton(sensor_vis, points_2d_cam2*scale, skeleton, (255, 0, 0), 1)   # 藍色：Groundtruth
    sensor_vis = draw_skeleton(sensor_vis, points_2d_MS, skeleton, (255, 0, 255), 1) # 紫色：M_Fuse
    # sensor_vis = draw_skeleton(sensor_vis, Cam1toCam2_kpts_proj, skeleton, (0, 255, 255)) # 黃色：M->S Fuse
    # 再畫 keypoints 點
    sensor_vis = draw_keypoints(sensor_vis, points_2d_cam2*scale, (255, 0, 0), 3)   # 藍色：Groundtruth
    sensor_vis = draw_keypoints(sensor_vis, points_2d_MS, (255, 0, 255), 3) # 紫色：M_Fuse
    # sensor_vis = draw_keypoints(sensor_vis, Cam1toCam2_kpts_proj, (0, 255, 255)) # 黃色：M->S Fuse
    if Show_all == True or Mean > 15:
        cv2.imshow("Sensor view(blue=Master, purple=Sensor)", sensor_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # ================================= 驗證mesh =================================
    
    mesh_cam1_padding = mesh_cam1*10  # 單位mm 
    mesh_cam2_padding = mesh_cam2*10  # 單位mm 

    # 接著以 R/T 將 cam2_padding 轉到 cam1 座標系，然後計算 MPJPE / PA-MPJPE
    # Master view座標轉到Sensor view座標
    mesh_cam1_to_cam2_padding = (mesh_cam1_padding - T )@ R.T  # (N, 3) 
    # 將 Sensor view座標轉到Master view座標
    mesh_cam2_to_cam1_padding = (mesh_cam2_padding @ R) + T # (N, 3) 

    c1, R1, T1 = umeyama(mesh_cam1_to_cam2_padding, mesh_cam2_padding)
    mesh_cam1_to_cam2_padding = mesh_cam1_to_cam2_padding.dot(c1*R1) + T1
    c1, R1, T1 = umeyama(mesh_cam2_to_cam1_padding, mesh_cam1_padding)
    mesh_cam2_to_cam1_padding = mesh_cam2_to_cam1_padding.dot(c1*R1) + T1

    cam2_pa_mpjpe = compute_pa_mpjpe(mesh_cam1_to_cam2_padding, mesh_cam2_padding)
    cam1_pa_mpjpe = compute_pa_mpjpe(mesh_cam2_to_cam1_padding, mesh_cam1_padding)
    if Show_all == True or Mean > 15:
        print(f"Mesh PA-MPJPE: {(cam1_pa_mpjpe+cam2_pa_mpjpe)/2 :.2f} mm")

