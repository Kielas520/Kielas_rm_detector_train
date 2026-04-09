import random
import cv2
import shutil
from pathlib import Path

def visualize_dataset(root_path: str, data_type: str = "train", if_color: bool = True):
    # 1. 初始化路径对象
    root = Path(root_path)
    type_dir = root / data_type
    
    # 定义输出文件夹路径
    output_dir = root / f"{data_type}_visualized_samples"
    
    if not type_dir.exists():
        print(f"错误: 找不到目录 {type_dir}")
        return

    # 刷新文件夹（清空旧数据）
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"已清理旧的输出文件夹: {output_dir.name}")

    # 创建新的输出文件夹
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义日志文件路径
    log_file_path = output_dir / "sample_log.txt"

    # 2. 收集所有图片和对应的标签路径
    all_data = []
    class_dirs = [d for d in type_dir.iterdir() if d.is_dir()]
    
    for class_dir in class_dirs:
        class_id = class_dir.name
        photo_dir = class_dir / "photos"
        label_dir = class_dir / "labels"
        
        if not photo_dir.exists():
            continue
            
        for img_path in photo_dir.glob("*.jpg"):
            label_path = label_dir / img_path.with_suffix('.txt').name
            all_data.append((img_path, label_path, class_id))
    
    if not all_data:
        print(f"在 {type_dir} 下未发现任何图片")
        return

    # 3. 抽样 12 张图片
    sample_size = min(12, len(all_data))
    sampled_data = random.sample(all_data, sample_size)
    
    # 初始化日志内容
    log_contents = [
        f"抽样数据分析日志 (共抽样 {sample_size} 张)\n",
        f"设置: if_color={if_color}\n",
        "-" * 50 + "\n"
    ]
    
    # 4. 遍历抽样数据进行绘制、记录并保存
    for img_path, label_path, class_id in sampled_data:
        img = cv2.imread(str(img_path))
        if img is None:
            log_contents.append(f"[{class_id}] {img_path.name}: 读取图片失败\n")
            continue
            
        log_entry = f"[{class_id}] {img_path.name}: "
        
        if not label_path.exists():
            log_entry += "未找到对应的标签文件\n"
        else:
            valid_targets = 0
            invalid_targets = 0
            target_issues = []
            
            with label_path.open('r') as f:
                lines = f.readlines()
                if not lines:
                    log_entry += "标签文件存在但内容为空\n"
                else:
                    for line_idx, line in enumerate(lines):
                        parts = list(map(float, line.split()))
                        
                        # 检查数据完整性并提取坐标
                        is_valid = False
                        if if_color:
                            if len(parts) >= 10:
                                is_valid = True
                                pts = [
                                    (int(parts[2]), int(parts[3])), (int(parts[4]), int(parts[5])), 
                                    (int(parts[6]), int(parts[7])), (int(parts[8]), int(parts[9]))
                                ]
                            else:
                                target_issues.append(f"目标{line_idx+1}数据不全(仅{len(parts)}个值,预期>=10)")
                        else:
                            if len(parts) >= 9:
                                is_valid = True
                                pts = [
                                    (int(parts[1]), int(parts[2])), (int(parts[3]), int(parts[4])), 
                                    (int(parts[5]), int(parts[6])), (int(parts[7]), int(parts[8]))
                                ]
                            else:
                                target_issues.append(f"目标{line_idx+1}数据不全(仅{len(parts)}个值,预期>=9)")
                        
                        if is_valid:
                            valid_targets += 1
                            # 绘制目标框和坐标点
                            cv2.line(img, pts[1], pts[0], (0, 255, 0), 2)
                            cv2.line(img, pts[3], pts[2], (0, 255, 0), 2)
                            for p in pts:
                                cv2.circle(img, p, 4, (255, 0, 0), -1)
                                coord_text = f"({p[0]},{p[1]})"
                                cv2.putText(img, coord_text, (p[0] + 5, p[1] - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                        else:
                            invalid_targets += 1
                            
                    # 汇总当前图像的日志
                    log_entry += f"正常目标数: {valid_targets}, 异常目标数: {invalid_targets}"
                    if target_issues:
                        log_entry += f" | 异常明细: {', '.join(target_issues)}"
                    log_entry += "\n"

        log_contents.append(log_entry)

        # 标注类别信息
        cv2.putText(img, f"Class: {class_id}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # 保存处理后的图像
        out_img_path = output_dir / f"{class_id}_{img_path.name}"
        cv2.imwrite(str(out_img_path), img)

    # 5. 将日志写入到文本文件
    with log_file_path.open('w', encoding='utf-8') as log_f:
        log_f.writelines(log_contents)

    print(f"成功抽样并处理 {sample_size} 张图片。")
    print(f"结果图片及日志已保存至: {output_dir.absolute()}")
    print(f"请查看 {log_file_path.name} 获取数据完整性记录。")

# 调用示例 (if_color=True 代表按照含有颜色标志位解析)
visualize_dataset("./data", "raw", if_color=True)