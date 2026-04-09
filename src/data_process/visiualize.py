import random
import cv2
import shutil
from pathlib import Path

def visualize_dataset(root_path: str, data_type: str = "train", if_flag: list = None):
    if if_flag is None:
        if_flag = [0, 0]
        
    root = Path(root_path)
    type_dir = root / data_type
    output_dir = root / "visualized_samples" / f"{data_type}"
    
    if not type_dir.exists():
        print(f"错误: 找不到目录 {type_dir}")
        return

    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"已清理旧的输出文件夹: {output_dir.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / "sample_log.txt"

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

    sample_size = min(12, len(all_data))
    sampled_data = random.sample(all_data, sample_size)
    
    has_flag = (if_flag[0] == 1)
    flag_type = if_flag[1] 
    
    log_contents = [
        f"抽样数据分析日志 (共抽样 {sample_size} 张)\n",
        f"设置: if_flag={if_flag} (是否包含标志位: {has_flag}, 标志位含义: {'颜色' if flag_type == 0 else '可见度' if flag_type == 1 else '无'})\n",
        "-" * 50 + "\n"
    ]
    
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
                        
                        is_valid = False
                        flag_val = None
                        
                        if has_flag:
                            if len(parts) >= 10:
                                is_valid = True
                                flag_val = int(parts[1])
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
                            box_color = (0, 255, 0)
                            info_text = ""
                            
                            if has_flag:
                                if flag_type == 0:
                                    if flag_val == 0:
                                        box_color = (0, 0, 255) 
                                        info_text = f"Red"
                                    elif flag_val == 1:
                                        box_color = (255, 0, 0)
                                        info_text = f"Blue"
                                    else:
                                        info_text = f"c:{flag_val}"
                                elif flag_type == 1:
                                    if flag_val == 0: info_text = "Inv"
                                    elif flag_val == 1: info_text = "Vague"
                                    elif flag_val == 2: info_text = "Full"
                                    else: info_text = f"s:{flag_val}"

                            # 1. 绘制目标线段 (左右垂直边缘)
                            cv2.line(img, pts[1], pts[0], box_color, 2)
                            cv2.line(img, pts[3], pts[2], box_color, 2)
                            
                            # 2. 绘制各角点坐标
                            for p in pts:
                                cv2.circle(img, p, 4, box_color, -1)
                                cv2.putText(img, f"({p[0]},{p[1]})", (p[0] + 5, p[1] - 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

                            # 3. 在每个目标左上方绘制 ID 和 状态信息
                            # 计算目标的左上角位置 (取所有点的最小x和最小y)
                            min_x = min(p[0] for p in pts)
                            min_y = min(p[1] for p in pts)
                            
                            target_label = f"id:{class_id} {info_text}" if info_text else f"id:{class_id}"
                            
                            # 绘制文字，位置稍微向上偏移 10 像素以防覆盖线段
                            cv2.putText(img, target_label, (min_x, min_y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
                        else:
                            invalid_targets += 1
                            
                    log_entry += f"正常目标数: {valid_targets}, 异常目标数: {invalid_targets}"
                    if target_issues:
                        log_entry += f" | 异常明细: {', '.join(target_issues)}"
                    log_entry += "\n"

        log_contents.append(log_entry)
        out_img_path = output_dir / f"{class_id}_{img_path.name}"
        cv2.imwrite(str(out_img_path), img)

    with log_file_path.open('w', encoding='utf-8') as log_f:
        log_f.writelines(log_contents)

    print(f"成功抽样并处理 {sample_size} 张图片。")
    print(f"结果图片及日志已保存至: {output_dir.absolute()}")

# 调用示例
if __name__ == "__main__":
    visualize_dataset("./data", "augment", if_flag=[1, 1])