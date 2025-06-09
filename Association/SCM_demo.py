import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
from tqdm import tqdm

# ==== 1. 解码函数：moment-based 点集转bbox ====
def moment_decode(points_batch):
    x = points_batch[:, :, 0]
    y = points_batch[:, :, 1]
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_std = x.std(dim=1, keepdim=True)
    y_std = y.std(dim=1, keepdim=True)

    width = 2 * x_std
    height = 2 * y_std
    x1 = x_mean - width / 2
    y1 = y_mean - height / 2
    x2 = x_mean + width / 2
    y2 = y_mean + height / 2
    return torch.cat([x1, y1, x2, y2], dim=1)

# ==== 2. SCM关联策略（带进度条） ====
def scm_association(pred_pois, det_pois, moment_decode, eps=1e-6):
    K_pred = len(pred_pois)
    K_det = len(det_pois)
    cost_matrix = np.zeros((K_pred, K_det), dtype=np.float32)

    for i in tqdm(range(K_pred), desc="SCM Matching"):
        P = pred_pois[i]
        box_P = moment_decode(P.unsqueeze(0))[0]
        area_P = (box_P[2] - box_P[0]) * (box_P[3] - box_P[1]) + eps

        for j in range(K_det):
            D = det_pois[j]
            box_D = moment_decode(D.unsqueeze(0))[0]
            area_D = (box_D[2] - box_D[0]) * (box_D[3] - box_D[1]) + eps

            match = 0
            for p in P:
                if (box_D[0] <= p[0] <= box_D[2]) and (box_D[1] <= p[1] <= box_D[3]):
                    match += 1
            match_ratio = match / (P.shape[0] + eps)
            w_ij = min(1.0, float(area_P / area_D))
            sim_ij = w_ij * match_ratio
            cost_matrix[i, j] = 1.0 - sim_ij
    return cost_matrix

# ==== 3. 仿真生成点集 ====
def simulate_pois(center, num_points=9, spread=2.0):
    x_center, y_center = center
    points = torch.randn(num_points, 2) * spread + torch.tensor([x_center, y_center])
    return points

# ==== 4. 可视化函数 ====
def draw_bbox(ax, box, color='k', linestyle='-'):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    rect = plt.Rectangle((x1, y1), width, height, linewidth=1.5, edgecolor=color,
                         facecolor='none', linestyle=linestyle)
    ax.add_patch(rect)

# ==== 5. 主程序入口 ====
def main():
    # 构造模拟的预测和检测点集
    pred_pois = [simulate_pois((10, 10)), simulate_pois((30, 30)), simulate_pois((50, 10))]
    det_pois = [simulate_pois((11, 10)), simulate_pois((29, 29)), simulate_pois((70, 15))]

    # 计算SCM代价矩阵
    cost_matrix = scm_association(pred_pois, det_pois, moment_decode)

    # 匈牙利算法进行匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = list(zip(row_ind, col_ind))

    # 可视化并保存图像
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b']

    # 画点集
    for i, poi in enumerate(pred_pois):
        poi_np = poi.numpy()
        ax.scatter(poi_np[:, 0], poi_np[:, 1], marker='o', color=colors[i], label=f'Pred {i}')
    for j, poi in enumerate(det_pois):
        poi_np = poi.numpy()
        ax.scatter(poi_np[:, 0], poi_np[:, 1], marker='x', color=colors[j], label=f'Det {j}')

    # 画bbox
    for i, poi in enumerate(pred_pois):
        box = moment_decode(poi.unsqueeze(0))[0].numpy()
        draw_bbox(ax, box, color=colors[i], linestyle='--')
    for j, poi in enumerate(det_pois):
        box = moment_decode(poi.unsqueeze(0))[0].numpy()
        draw_bbox(ax, box, color=colors[j], linestyle=':')

    # 连线
    for i, j in matches:
        pred_center = pred_pois[i].mean(dim=0)
        det_center = det_pois[j].mean(dim=0)
        ax.plot([pred_center[0], det_center[0]], [pred_center[1], det_center[1]], '--k')

    ax.legend()
    ax.set_title("SCM PoI Matching with BBoxes")
    plt.axis('equal')
    plt.grid(True)
    fig.tight_layout()
    plt.savefig("scm_matching_result.png")
    plt.close(fig)

    # 输出代价矩阵
    df = pd.DataFrame(
        cost_matrix,
        columns=[f'Det_{i}' for i in range(len(det_pois))],
        index=[f'Pred_{i}' for i in range(len(pred_pois))]
    )
    print("\nSCM Cost Matrix:")
    print(df)

if __name__ == "__main__":
    main()
