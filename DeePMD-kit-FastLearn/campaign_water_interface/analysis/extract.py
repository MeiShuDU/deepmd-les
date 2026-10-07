import re
import csv

LOG_FILE = "/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_fastlearn/cace/runs/cace-sea-lr/train.log"
OUT_FILE = "/root/app/deepmd-les/deepmd-les/DeePMD-kit-FastLearn/campaign_fastlearn/cace/runs/cace-sea-lr/val_rmse.csv"

# 匹配 Epoch 行，例如：
# Epoch 1, Train Loss: 488.9213, Val Loss: 44.4994
epoch_re = re.compile(r"Epoch\s+(\d+),\s*Train Loss")

# 匹配需要提取的两个指标
val_e_re = re.compile(r"val_e/atom_rmse:\s*([0-9.eE+-]+)")
val_f_re = re.compile(r"val_f_rmse:\s*([0-9.eE+-]+)")

records = []
cur = None

with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        # 遇到新的 Epoch 行
        m = epoch_re.search(line)
        if m:
            epoch = int(m.group(1))

            # 如果是同一个 epoch 的重复打印行，跳过
            if cur is not None and cur["epoch"] == epoch:
                continue

            # 上一个 epoch 如果已经收齐两个指标，就保存
            if cur is not None and cur["val_e_atom_rmse"] is not None and cur["val_f_rmse"] is not None:
                records.append(cur)

            # 开始记录当前 epoch
            cur = {
                "epoch": epoch,
                "val_e_atom_rmse": None,
                "val_f_rmse": None,
            }
            continue

        if cur is None:
            continue

        # 提取 val_e/atom_rmse
        m = val_e_re.search(line)
        if m and cur["val_e_atom_rmse"] is None:
            cur["val_e_atom_rmse"] = float(m.group(1))
            continue

        # 提取 val_f_rmse
        m = val_f_re.search(line)
        if m and cur["val_f_rmse"] is None:
            cur["val_f_rmse"] = float(m.group(1))

# 保存最后一个 epoch
if cur is not None and cur["val_e_atom_rmse"] is not None and cur["val_f_rmse"] is not None:
    records.append(cur)

# 写入 CSV
with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["epoch", "val_e_atom_rmse", "val_f_rmse"]
    )
    writer.writeheader()
    writer.writerows(records)

print(f"共提取 {len(records)} 条记录，已写入 {OUT_FILE}")

# 打印前几行看看
for r in records[:10]:
    print(r)